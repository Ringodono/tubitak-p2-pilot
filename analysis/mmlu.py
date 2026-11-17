#!/usr/bin/env python3
"""Pilot 4: KL kazançları görev doğruluğuna yansıyor mu? (TurkishMMLU, sıfır atış)

Pilot 3'teki nöron ayırma benzetimi (keep.py) aynen kullanılır: model kuantize GGUF'tan
yüklenir, seçilen FFN nöronları F16 değerlerine döndürülür. Her soru için yalnız son
konumun logitleri hesaplanır ve " A".." E" harfleri karşılaştırılır.

Kollar (ETİKET=kuantize.gguf:seçim, seçim: none | err_en_x1 | err_tr_x1 | ...):
  f16 için dosya yerine "f16" yazılır.

Ölçütler: doğruluk, doğru harfin olasılığı (log), F16 ile aynı cevap oranı ve
seçenekler üzerindeki dağılımın F16'ya göre KL sapması. Karşılaştırmalar soru düzeyinde
eşleştirilmiştir (bootstrap ve kesin McNemar).

Kullanım: python3 analysis/mmlu.py --hf DİZİN --gguf F16.gguf --data data/turkishmmlu \
            [--n 450] f16=f16:none q3_en=keep/q3_k_s_en.gguf:none ...
Çıktı: $RES/mmlu.npz (kol başına soru düzeyinde harf log olasılıkları); rapor: report4.py
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import RES, device, load_gguf_weights, load_hf, load_neurons
from keep import FFN, top_global

LETTERS = "ABCDE"
PROMPT = "Aşağıdaki çoktan seçmeli soruyu yanıtlayın.\n\nSoru: {q}\n{opts}\nCevap:"


def load_questions(data_dir, n, seed=0):
    """Ders başına eşit sayıda soru (tabakalı, sabit tohum)."""
    files = sorted(Path(data_dir).glob("*.json"))
    if not files:
        raise SystemExit(f"Soru dosyası yok: {data_dir}")
    per = max(1, n // len(files))
    rng = np.random.default_rng(seed)
    out = []
    for f in files:
        rows = [r for r in json.loads(f.read_text()) if 2 <= len(r["choices"]) <= 5 and r["answer"] in LETTERS]
        idx = rng.permutation(len(rows))[:per]
        subj = f.stem.replace("TurkishMMLU_", "")
        out += [dict(rows[i], subject=subj) for i in sorted(idx)]
    return out


def build_prompt(r):
    opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(r["choices"]))
    return PROMPT.format(q=r["question"].strip(), opts=opts)


def letter_logprobs(model, prompts, letter_ids, dev):
    """Her istem için seçenek harflerinin log olasılıkları (yalnız seçenekler arasında)."""
    out = []
    with torch.inference_mode():
        for ids, k in prompts:
            h = model.model(torch.tensor([ids], device=dev)).last_hidden_state[:, -1]
            logits = model.lm_head(h)[0].float()
            out.append(torch.log_softmax(logits[letter_ids[:k]], dim=-1).cpu().numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True, help="F16 GGUF")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("arms", nargs="+", help="ETİKET=dosya.gguf:seçim")
    args = ap.parse_args()
    dev = device()
    torch.set_grad_enabled(False)

    qs = load_questions(args.data, args.n)
    print(f"{len(qs)} soru, {len({q['subject'] for q in qs})} ders")

    tok, model = load_hf(args.hf, dev, args.gguf)
    layers = model.model.layers
    letter_ids = []
    for L in LETTERS:
        t = tok(" " + L, add_special_tokens=False)["input_ids"]
        if len(t) != 1:
            raise SystemExit(f"' {L}' tek token değil: {t}")
        letter_ids.append(t[0])
    letter_ids = torch.tensor(letter_ids, device=dev)
    prompts = [(tok(build_prompt(q), add_special_tokens=False)["input_ids"], len(q["choices"])) for q in qs]
    gold = np.array([LETTERS.index(q["answer"]) for q in qs])
    print(f"ortalama istem uzunluğu: {np.mean([len(p) for p, _ in prompts]):.0f} token")

    nd = load_neurons()
    nlangs = list(nd["langs"])
    tr_mask = nd["masks"][nlangs.index("tr")]
    a2 = nd["a2"].astype(np.float64)
    K = int(tr_mask.sum())
    orig = [{k: getattr(l.mlp, k).weight.detach().to("cpu", torch.float16).clone() for k in FFN} for l in layers]
    wnorm_down = np.stack([(orig[i]["down_proj"].float() ** 2).sum(0).numpy() for i in range(len(layers))])

    out_npz = RES / "mmlu.npz"
    res = dict(np.load(out_npz, allow_pickle=True)) if out_npz.exists() else {}
    res["gold"] = gold
    res["subject"] = np.array([q["subject"] for q in qs])
    loaded = "f16"
    t_start = time.time()
    for n_done, spec in enumerate(args.arms):
        label, rest = spec.split("=", 1)
        path, sel_name = rest.rsplit(":", 1)
        key = f"{label}|{sel_name}"
        if key in res and len(res[key]) == len(qs):
            print(f"== {key}: tamamlanmış")
            continue
        t0 = time.time()
        if path != loaded:
            load_gguf_weights(model, args.gguf if path == "f16" else path)
            loaded = path
        saved = []
        if sel_name != "none":
            lang, m = sel_name[4:].split("_x")
            dd, rel = [], []
            for i, l in enumerate(layers):
                w = {k: getattr(l.mlp, k).weight.detach().cpu() for k in FFN}
                o = {k: orig[i][k].float() for k in FFN}
                dd.append(((w["down_proj"] - o["down_proj"]) ** 2).sum(0).numpy())
                rel.append(sum((((w[k] - o[k]) ** 2).sum(1) / (o[k] ** 2).sum(1).clamp_min(1e-12)).numpy()
                               for k in ("gate_proj", "up_proj")))
            score = a2[nlangs.index(lang)] * (np.stack(dd) + wnorm_down * np.stack(rel))
            sel = top_global(score, int(round(float(m) * K)))
            for i, l in enumerate(layers):
                if not sel[i].any():
                    continue
                idx = torch.from_numpy(np.flatnonzero(sel[i]))
                for k in FFN:
                    p = getattr(l.mlp, k).weight
                    j = idx.to(p.device)
                    if k == "down_proj":
                        saved.append((p, k, j, p.data[:, j].clone()))
                        p.data[:, j] = orig[i][k][:, idx].float().to(p.device)
                    else:
                        saved.append((p, k, j, p.data[j].clone()))
                        p.data[j] = orig[i][k][idx].float().to(p.device)
            print(f"{key}: {int(sel.sum())} nöron ayrıldı")
        lp = letter_logprobs(model, prompts, letter_ids, dev)
        for p, k, j, v in saved:  # kuantize değerlere geri dön
            if k == "down_proj":
                p.data[:, j] = v
            else:
                p.data[j] = v
        mat = np.full((len(qs), len(LETTERS)), -np.inf, dtype=np.float32)
        for r, v in enumerate(lp):
            mat[r, :len(v)] = v
        res[key] = mat
        np.savez_compressed(out_npz, **res)
        acc = (mat.argmax(1) == gold).mean()
        el = time.time() - t0
        left = len(args.arms) - n_done - 1
        print(f"{key:28s} doğruluk {acc:.3f}  ({el / 60:.1f} dk; kalan kol {left}, tahmini {left * el / 60:.0f} dk)",
              flush=True)
    print(f"toplam {(time.time() - t_start) / 60:.0f} dk -> {out_npz}")


if __name__ == "__main__":
    main()
