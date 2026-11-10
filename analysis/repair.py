#!/usr/bin/env python3
"""M1 nedensel testi ve M3 token türü analizi.

Modelin yalnız FFN ağırlıkları (gate, up, down) kuantize sürümle değiştirilir;
dikkat katmanları F16 kalır. Ardından bir nöron kümesi asıl ağırlıklarına döndürülür
(gate ve up satırı, down sütunu) ve F16 modele göre KL sapması ölçülür.
Kümeler: dile özgü nöronlar (tr, en, fi, ...) ve TR kadar rastgele nöron (birkaç tohum).
TR nöronlarını onarmak Türkçe KLD'yi rastgele onarımdan belirgin biçimde fazla
düşürüyorsa, Türkçe kaybın bu nöronlardan geçtiğine nedensel kanıttır.

M3: "quant" yapılandırmasında token başına KLD, tahmin edilen tokenin türüne göre
(kelime başı / kelime içi devam / diğer) ayrıştırılır.

Kullanım: python3 analysis/repair.py --hf DİZİN --quant quant.gguf [--langs tr,en,fi] [--chunks 16] [--seeds 3]
Çıktı: $RES/repair.csv, $RES/repair_tokens.npz
"""
import argparse
import csv
import time

import numpy as np
import torch

from common import RES, Gguf, chunks, device, load_hf, load_neurons, random_like


def token_classes(tok, ids):
    """0 = kelime başı (önünde boşluk işareti), 1 = kelime içi devam, 2 = diğer."""
    out = np.full(len(ids), 2, dtype=np.int8)
    for i, piece in enumerate(tok.convert_ids_to_tokens(list(ids))):
        if piece is None:
            continue
        if piece[0] in ("Ġ", "▁") and len(piece) > 1 and (piece[1].isalpha() or ord(piece[1]) > 127):
            out[i] = 0
        elif piece[0].isalpha() or ord(piece[0]) > 127:
            out[i] = 1
    return out


def logprobs(model, ids, dev):
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=dev)).logits[0].float()
    half = len(ids) // 2
    # llama-perplexity gibi: pencerenin ikinci yarısında bir sonraki token dağılımı
    return torch.log_softmax(logits[half:-1], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="yapılandırma ve tokenizer için Hugging Face dizini")
    ap.add_argument("--gguf", required=True, help="ağırlıkların alınacağı F16 GGUF (llama.cpp ölçümleriyle aynı model)")
    ap.add_argument("--quant", required=True)
    ap.add_argument("--langs", default="tr,en,fi")
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    langs = args.langs.split(",")
    dev = device()
    tok, model = load_hf(args.hf, dev, args.gguf)
    layers = model.model.layers
    nd = load_neurons()
    nlangs = list(nd["langs"])
    masks = nd["masks"]

    data = {lang: chunks(tok, lang, args.chunks) for lang in langs}
    t0 = time.time()
    base = {lang: [logprobs(model, ids, dev).half().cpu() for ids in data[lang]] for lang in langs}
    print(f"taban logitleri: {time.time() - t0:.0f} sn")

    orig = [{k: getattr(l.mlp, k).weight.detach().cpu().clone() for k in ("gate_proj", "up_proj", "down_proj")} for l in layers]
    q = Gguf(args.quant)
    quant = [{"gate_proj": torch.from_numpy(q.get(f"blk.{i}.ffn_gate.weight")),
              "up_proj": torch.from_numpy(q.get(f"blk.{i}.ffn_up.weight")),
              "down_proj": torch.from_numpy(q.get(f"blk.{i}.ffn_down.weight"))} for i in range(len(layers))]

    def set_ffn(restore):
        for i, layer in enumerate(layers):
            w = {k: v.clone() for k, v in quant[i].items()}
            if restore is not None and restore[i].any():
                idx = torch.from_numpy(np.flatnonzero(restore[i]))
                w["gate_proj"][idx] = orig[i]["gate_proj"][idx]
                w["up_proj"][idx] = orig[i]["up_proj"][idx]
                w["down_proj"][:, idx] = orig[i]["down_proj"][:, idx]
            for k, v in w.items():
                getattr(layer.mlp, k).weight.data.copy_(v.to(dev))

    configs = [("quant", None)]
    for lang in nlangs:
        configs.append((f"restore_{lang}", masks[nlangs.index(lang)]))
    for s in range(args.seeds):
        configs.append((f"restore_rand{s}", random_like(masks[nlangs.index("tr")], masks.any(axis=0), seed=s)))

    rows, per_token = [], {}
    for name, restore in configs:
        set_ffn(restore)
        n_rest = int(restore.sum()) if restore is not None else 0
        for lang in langs:
            kl = []
            for ids, lp0 in zip(data[lang], base[lang]):
                lp = logprobs(model, ids, dev).cpu()
                p0 = lp0.float()
                kl.append((p0.exp() * (p0 - lp)).sum(dim=-1).numpy())
            kl = np.concatenate(kl)
            per_token[f"{name}|{lang}"] = kl.astype(np.float32)
            rows.append({"config": name, "lang": lang, "n_restored": n_rest, "kld": float(kl.mean()),
                         "kld_se": float(kl.std() / np.sqrt(len(kl)))})
            print(f"{name:14s} {lang} KLD {kl.mean():.5f} (onarılan nöron: {n_rest})")

    # M3: tahmin edilen (bir sonraki) tokenin türü
    for lang in langs:
        nxt = np.concatenate([np.array(ids[len(ids) // 2 + 1:]) for ids in data[lang]])
        per_token[f"class|{lang}"] = token_classes(tok, nxt)

    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "repair.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(RES / "repair_tokens.npz", **per_token)
    print(f"{time.time() - t0:.0f} sn -> {RES / 'repair.csv'}")


if __name__ == "__main__":
    main()
