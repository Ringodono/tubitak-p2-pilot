#!/usr/bin/env python3
"""M1/M2: kuantizasyon hatasının FFN nöronlarına dağılımı.

Her kuantize model ve katman için, nöron j'nin göreli ağırlık hatası
  e_j = ||ΔW_down[:, j]||² + ||ΔW_gate[j, :]||² + ||ΔW_up[j, :]||²  /  aynısı W için
ve dil L için çıktı hata enerjisi  E_L = Σ_j a2_L[j] · ||ΔW_down[:, j]||²  hesaplanır.
Kümeler: dile özgü nöronlar (her dil), rastgele kontrol (TR kadar) ve tümü.

Kullanım: python3 analysis/qerror.py --f16 F16.gguf ETİKET=quant.gguf [...]
Çıktı: $RES/qerror.csv (model, katman, küme, n, göreli hata ortalaması, dil başına enerji payı)
"""
import argparse
import csv

import numpy as np

from common import RES, Gguf, load_neurons, random_like


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f16", required=True)
    ap.add_argument("quant", nargs="+", help="ETİKET=dosya.gguf")
    args = ap.parse_args()
    nd = load_neurons()
    langs = list(nd["langs"])
    masks, a2 = nd["masks"], nd["a2"]
    sets = {lang: masks[d] for d, lang in enumerate(langs)}
    sets["rand"] = random_like(sets["tr"], masks.any(axis=0), seed=0)
    base = Gguf(args.f16)
    n_layers = base.n_layers()

    rows = []
    for spec in args.quant:
        label, path = spec.split("=", 1)
        q = Gguf(path)
        print(f"== {label}")
        for layer in range(n_layers):
            name = lambda t: f"blk.{layer}.{t}.weight"
            w = {t: base.get(name(t)) for t in ("ffn_gate", "ffn_up", "ffn_down")}
            dw = {t: q.get(name(t)) - w[t] for t in w}
            err_down = (dw["ffn_down"] ** 2).sum(axis=0)
            err = err_down + (dw["ffn_gate"] ** 2).sum(axis=1) + (dw["ffn_up"] ** 2).sum(axis=1)
            ref = (w["ffn_down"] ** 2).sum(axis=0) + (w["ffn_gate"] ** 2).sum(axis=1) + (w["ffn_up"] ** 2).sum(axis=1)
            rel = err / np.clip(ref, 1e-12, None)
            energy = {lang: a2[d, layer] * err_down for d, lang in enumerate(langs)}
            for set_name, m in list(sets.items()) + [("all", np.ones_like(masks[0]))]:
                sel = m[layer]
                if not sel.any():
                    continue
                row = {"model": label, "layer": layer, "set": set_name, "n": int(sel.sum()),
                       "rel_err": float(rel[sel].mean())}
                for lang in langs:
                    row[f"energy_share_{lang}"] = float(energy[lang][sel].sum() / max(energy[lang].sum(), 1e-30))
                rows.append(row)

    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "qerror.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} satır -> {RES / 'qerror.csv'}")


if __name__ == "__main__":
    main()
