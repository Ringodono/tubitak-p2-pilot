#!/usr/bin/env python3
"""Y2: dile özgü Fisher bilgisiyle kuantizasyon hassasiyetinin ucuz kestirimi.

İkinci dereceden yaklaşım: bir grubun ağırlıkları δ kadar bozulduğunda dil L için
beklenen KL sapması  KL_L ≈ ½ Σ F_L ⊙ δ².  F_L, dil L metninde log-olabilirlik
gradyanlarının karesinin ortalaması (empirik Fisher köşegeni); δ, saf Q2_K
kuantizasyonun F16'dan farkı. Tahmin, profilde ölçülen KLD ile (her grup Q2_K)
report2.py'de karşılaştırılır. Ölçüm 56 kuantizasyon gerektirirken bu kestirim
dil başına birkaç ileri-geri geçiş gerektirir.

Kullanım: python3 analysis/fisher.py --hf DİZİN --q2 saf_q2k.gguf [--langs tr,en,fi] [--chunks 16]
Çıktı: $RES/fisher.csv (layer, group, lang, est)
"""
import argparse
import csv
import time

import torch

from common import GROUPS, HF_NAMES, RES, Gguf, chunks, device, load_hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="yapılandırma ve tokenizer için Hugging Face dizini")
    ap.add_argument("--gguf", required=True, help="ağırlıkların alınacağı F16 GGUF (llama.cpp ölçümleriyle aynı model)")
    ap.add_argument("--q2", required=True)
    ap.add_argument("--langs", default="tr,en,fi")
    ap.add_argument("--chunks", type=int, default=16)
    args = ap.parse_args()
    dev = device()
    tok, model = load_hf(args.hf, dev, args.gguf)
    layers = model.model.layers
    for p in model.parameters():
        p.requires_grad_(False)
    targets = {}
    for i, layer in enumerate(layers):
        for g, tensors in GROUPS.items():
            for t in tensors:
                mod = layer.get_submodule(HF_NAMES[t])
                mod.weight.requires_grad_(True)
                targets[(i, g, t)] = mod.weight

    q2 = Gguf(args.q2)

    rows = []
    for lang in args.langs.split(","):
        t0 = time.time()
        acc = {k: torch.zeros_like(w, device="cpu") for k, w in targets.items()}
        data = chunks(tok, lang, args.chunks, kind="calib")
        for ids in data:
            model.zero_grad(set_to_none=True)
            x = torch.tensor([ids], device=dev)
            out = model(x, labels=x)
            n = len(ids) - 1
            # Toplam kaybın gradyanının karesi / n: tokenler bağımsızsa token başına Fisher'ın yansız kestirimi
            (out.loss * n).backward()
            for k, w in targets.items():
                acc[k] += w.grad.detach().cpu() ** 2 / n
        est = {}
        for (i, g, t), a in acc.items():
            # δ² her seferinde yeniden açılır: tüm δ'leri bellekte tutmak ~model boyutu kadar RAM ister
            d2 = (torch.from_numpy(q2.get(f"blk.{i}.{t}.weight")) - targets[(i, g, t)].detach().cpu()) ** 2
            est[(i, g)] = est.get((i, g), 0.0) + 0.5 * float((a / len(data) * d2).sum())
        for (i, g), v in sorted(est.items()):
            rows.append({"layer": i, "group": g, "lang": lang, "est": v})
        print(f"{lang}: {len(data)} parça, {time.time() - t0:.0f} sn")

    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "fisher.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {RES / 'fisher.csv'}")


if __name__ == "__main__":
    main()
