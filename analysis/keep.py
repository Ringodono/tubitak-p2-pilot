#!/usr/bin/env python3
"""Pilot 3 / Y1': Türkçe nöronları yüksek hassasiyette ayırma (nöron düzeyinde karışık hassasiyet).

Tüm model (dikkat, FFN, gömme) llama.cpp ile kuantize edilmiş GGUF'tan yüklenir. Ardından
seçilen FFN nöronlarının ağırlıkları (gate ve up satırı, down sütunu) F16 (ya da Q8_0)
değerlerine döndürülür ve F16 modele göre KL sapması ölçülür. Bu, nöronları ayrı bir
yüksek hassasiyetli tabloda saklayan bir biçimin (SpQR / aykırı değer koruma benzeri)
benzetimidir; ek boyut her yapılandırma için hesaplanır.

Seçimler (K = Türkçe LAPE nöron sayısı, xM = M*K nöron):
  none          ayırma yok (yalnız kuantize model)
  lape_tr       Türkçe LAPE nöronları, F16            lape_tr_q8  aynısı, Q8_0
  rand{s}       TR'nin katman dağılımında rastgele nöronlar (dile özgü nöronlar hariç)
  imp_{en,tr}   en büyük a2_L * ||W_down[:,j]||² (aktivasyon ağırlıklı katkı), L diliyle
  err_{en,tr}_xM  en büyük aktivasyon ağırlıklı kuantizasyon hatası, L diliyle:
                a2_L[j] * (||ΔW_down[:,j]||² + ||W_down[:,j]||² * (göreli hata gate_j + up_j))
                (err_en = dil bilgisi kullanmayan standart aykırı değer koruma)
  hyb_xM        lape_tr + kalan bütçe err_en ile (dile duyarlı + standart)
Kapsam: full = hepsi, core = none, lape_tr, rand0, imp_tr, err_en_x1, err_tr_x1, err_en_x2, hyb_x2;
        none = yalnız ayırmasız (ör. katman yükseltme GGUF'ları).

Kullanım: python3 analysis/keep.py --hf DİZİN --gguf F16.gguf [--langs tr,en] [--chunks 16] \
            ETİKET=quant.gguf@full ETİKET2=quant2.gguf@core ...
Çıktı: $RES/keep.csv, $RES/keep_tokens.npz
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from common import RES, Gguf, chunks, device, load_gguf_weights, load_hf, load_neurons, random_like

FFN = ("gate_proj", "up_proj", "down_proj")
GG = {"gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down"}
CORE = ["none", "lape_tr", "rand0", "imp_tr", "err_en_x1", "err_tr_x1", "err_en_x2", "hyb_x2"]
FULL = ["none", "lape_tr", "lape_tr_q8", "rand0", "rand1", "rand2", "imp_en", "imp_tr",
        *[f"err_{l}_x{m}" for l in ("en", "tr") for m in ("0.5", "1", "2", "4")], "hyb_x2", "hyb_x4"]


def logprobs(model, ids, dev):
    with torch.inference_mode():
        logits = model(torch.tensor([ids], device=dev)).logits[0].float()
    half = len(ids) // 2
    return torch.log_softmax(logits[half:-1], dim=-1)


def q8(x):
    """Q8_0 benzetimi: 32'lik bloklar, blok başına fp16 ölçek."""
    shape = x.shape
    b = x.float().reshape(-1, 32)
    d = (b.abs().amax(dim=1, keepdim=True) / 127).half().float()
    q = torch.where(d > 0, torch.round(b / d.clamp_min(1e-30)), torch.zeros_like(b))
    return (q * d).reshape(shape)


def top_global(score, k, exclude=None):
    """score: (katman, nöron). Tüm katmanlarda en büyük k skoru seçer."""
    s = score.copy()
    if exclude is not None:
        s[exclude] = -np.inf
    flat = np.argsort(s, axis=None)[::-1][:k]
    m = np.zeros(s.shape, dtype=bool)
    m.flat[flat] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True, help="F16 GGUF")
    ap.add_argument("--langs", default="tr,en")
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("quant", nargs="+", help="ETİKET=dosya.gguf@full|core|none")
    args = ap.parse_args()
    langs = args.langs.split(",")
    dev = device()
    torch.set_grad_enabled(False)

    nd = load_neurons()
    nlangs = list(nd["langs"])
    masks, a2 = nd["masks"], nd["a2"].astype(np.float64)
    tr_mask = masks[nlangs.index("tr")]
    K = int(tr_mask.sum())
    print(f"Türkçe LAPE nöronları: K = {K}")

    tok, model = load_hf(args.hf, dev, args.gguf)
    layers = model.model.layers
    hidden = model.config.hidden_size
    data = {lang: chunks(tok, lang, args.chunks) for lang in langs}
    t0 = time.time()
    base = {lang: [logprobs(model, ids, dev).half().cpu() for ids in data[lang]] for lang in langs}
    print(f"F16 taban logitleri: {time.time() - t0:.0f} sn")
    orig = [{k: getattr(l.mlp, k).weight.detach().to("cpu", torch.float16).clone() for k in FFN} for l in layers]
    wnorm_down = np.stack([(orig[i]["down_proj"].float() ** 2).sum(0).numpy() for i in range(len(layers))])

    out_csv = RES / "keep.csv"
    done = set()
    if out_csv.exists():
        with out_csv.open() as f:
            done = {(r["quant"], r["config"], r["lang"]) for r in csv.DictReader(f)}
    tok_path = RES / "keep_tokens.npz"
    per_token = dict(np.load(tok_path)) if tok_path.exists() else {}
    fields = ["quant", "config", "lang", "n_kept", "file_mb", "extra_mb", "total_mb", "kld", "kld_se"]

    for spec in args.quant:
        label, rest = spec.split("=", 1)
        path, scope = rest.rsplit("@", 1)
        configs = {"full": FULL, "core": CORE, "none": ["none"]}[scope]
        todo = [c for c in configs if any((label, c, l) not in done for l in langs)]
        if not todo:
            print(f"== {label}: tamamlanmış")
            continue
        print(f"== {label} ({path}): {len(todo)} yapılandırma")
        load_gguf_weights(model, path)
        g = Gguf(path)
        file_mb = os.path.getsize(path) / 2**20
        # nöron başına kuantize bayt (her FFN tensörü)
        n_inter = orig[0]["gate_proj"].shape[0]
        qbytes = np.array([[g.tensors[f"blk.{i}.{GG[k]}.weight"].n_bytes / n_inter for k in FFN] for i in range(len(layers))])

        # kuantizasyon hatası istatistikleri
        dd, rel = [], []
        for i, l in enumerate(layers):
            w = {k: getattr(l.mlp, k).weight.detach().cpu() for k in FFN}
            o = {k: orig[i][k].float() for k in FFN}
            dd.append(((w["down_proj"] - o["down_proj"]) ** 2).sum(0).numpy())
            r = sum((((w[k] - o[k]) ** 2).sum(1) / (o[k] ** 2).sum(1).clamp_min(1e-12)).numpy() for k in ("gate_proj", "up_proj"))
            rel.append(r)
        dd, rel = np.stack(dd), np.stack(rel)

        def err(lang):
            return a2[nlangs.index(lang)] * (dd + wnorm_down * rel)

        def imp(lang):
            return a2[nlangs.index(lang)] * wnorm_down

        def select(c):
            if c == "none":
                return None, "f16"
            if c == "lape_tr":
                return tr_mask, "f16"
            if c == "lape_tr_q8":
                return tr_mask, "q8"
            if c.startswith("rand"):
                return random_like(tr_mask, masks.any(axis=0), seed=int(c[4:])), "f16"
            if c.startswith("imp_"):
                return top_global(imp(c[4:]), K), "f16"
            if c.startswith("err_"):
                lang, m = c[4:].split("_x")
                return top_global(err(lang), int(round(float(m) * K))), "f16"
            if c.startswith("hyb_x"):
                extra = top_global(err("en"), int(round(float(c[5:]) * K)) - K, exclude=tr_mask)
                return tr_mask | extra, "f16"
            raise ValueError(c)

        for c in todo:
            sel, prec = select(c)
            saved = []
            n_kept, extra_b = 0, 0.0
            if sel is not None:
                hi = {"f16": 2.0, "q8": 34 / 32}[prec] * hidden
                for i, l in enumerate(layers):
                    if not sel[i].any():
                        continue
                    idx = torch.from_numpy(np.flatnonzero(sel[i]))
                    n_kept += len(idx)
                    extra_b += len(idx) * (sum(hi - qbytes[i]) + 4)  # 4 bayt: nöron indeksi
                    for k in FFN:
                        p = getattr(l.mlp, k).weight
                        o = orig[i][k].float()
                        if k == "down_proj":
                            saved.append((p, k, idx, p.data[:, idx.to(p.device)].clone()))
                            v = o[:, idx]
                            p.data[:, idx.to(p.device)] = (q8(v.T).T if prec == "q8" else v).to(p.device)
                        else:
                            saved.append((p, k, idx, p.data[idx.to(p.device)].clone()))
                            v = o[idx]
                            p.data[idx.to(p.device)] = (q8(v) if prec == "q8" else v).to(p.device)
            new_rows = []
            for lang in langs:
                if (label, c, lang) in done:
                    continue
                kl = []
                for ids, lp0 in zip(data[lang], base[lang]):
                    lp = logprobs(model, ids, dev).cpu()
                    p0 = lp0.float()
                    kl.append((p0.exp() * (p0 - lp)).sum(dim=-1).numpy())
                kl = np.concatenate(kl)
                per_token[f"{label}|{c}|{lang}"] = kl.astype(np.float32)
                new_rows.append({"quant": label, "config": c, "lang": lang, "n_kept": n_kept,
                                 "file_mb": f"{file_mb:.1f}", "extra_mb": f"{extra_b / 2**20:.2f}",
                                 "total_mb": f"{file_mb + extra_b / 2**20:.1f}", "kld": f"{kl.mean():.6f}",
                                 "kld_se": f"{kl.std() / np.sqrt(len(kl)):.6f}"})
                print(f"{label:16s} {c:12s} {lang} KLD {kl.mean():.5f}  (+{extra_b / 2**20:.1f} MB, {n_kept} nöron)", flush=True)
            # geri al: kuantize değerler
            for p, k, idx, v in saved:
                if k == "down_proj":
                    p.data[:, idx.to(p.device)] = v
                else:
                    p.data[idx.to(p.device)] = v
            RES.mkdir(parents=True, exist_ok=True)
            new_file = not out_csv.exists()
            with out_csv.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if new_file:
                    w.writeheader()
                w.writerows(new_rows)
            np.savez_compressed(tok_path, **per_token)
            done |= {(label, c, l) for l in langs}
    print(f"{time.time() - t0:.0f} sn -> {out_csv}")


if __name__ == "__main__":
    main()
