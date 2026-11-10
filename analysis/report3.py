#!/usr/bin/env python3
"""Pilot 3 raporu: results/<model>/keep.csv -> results/pilot3_report.md

Kullanım: python3 analysis/report3.py [results]
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "results")

# Eşleştirilmiş karşılaştırmalar (aynı kuantize taban, Türkçe tokenlar): a - b
PAIRS = [("lape_tr", "rand*", "TR nöronları vs aynı sayıda rastgele"),
         ("lape_tr", "imp_tr", "TR nöronları vs TR aktivasyon katkısı (aynı sayı)"),
         ("lape_tr", "err_en_x1", "TR nöronları vs standart hata tabanlı seçim (aynı sayı)"),
         ("lape_tr", "err_tr_x1", "TR nöronları vs TR hata tabanlı seçim (aynı sayı)"),
         ("hyb_x2", "err_en_x2", "TR nöronları + standart vs yalnız standart (2K)"),
         ("hyb_x2", "err_tr_x2", "TR nöronları + standart vs TR hata tabanlı (2K)"),
         ("lape_tr_q8", "lape_tr", "TR nöronları Q8_0 vs F16")]


def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def section(tag, d):
    path = d / "keep.csv"
    if not path.exists():
        return []
    with path.open() as f:
        rs = list(csv.DictReader(f))
    tab = {}
    for r in rs:
        e = tab.setdefault((r["quant"], r["config"]), {"n": r["n_kept"], "extra": float(r["extra_mb"]), "total": float(r["total_mb"])})
        e[r["lang"]] = float(r["kld"])
    quants = list(dict.fromkeys(q for q, _ in tab))
    langs = [l for l in ("tr", "en", "fi") if any(l in v for v in tab.values())]
    L = [f"## {tag}", ""]

    # 1. Her taban için yapılandırmalar
    for q in [q for q in quants if "+up_" not in q]:
        ref = tab.get((q, "none"))
        L += [f"### {q}", "", "| Seçim | Nöron | +MB | Toplam MB | " + " | ".join(f"KLD {l} (değişim)" for l in langs) + " |",
              "| --- | --- | --- | --- | " + " | ".join("---" for _ in langs) + " |"]
        for (qq, c), v in tab.items():
            if qq != q:
                continue
            cells = []
            for l in langs:
                if l not in v:
                    cells.append("-")
                    continue
                ch = f" ({(v[l] / ref[l] - 1) * 100:+.1f}%)" if ref and l in ref and c != "none" else ""
                cells.append(f"{v[l]:.4f}{ch}")
            L.append(f"| {c} | {v['n']} | {v['extra']:.1f} | {v['total']:.1f} | " + " | ".join(cells) + " |")
        L.append("")

    # 2. Eşit boyut: katman yükseltme vs nöron ayırma (Pareto)
    ups = [q for q in quants if "+up_" in q]
    if ups:
        L += ["### Boyut–kalite: nöron ayırma vs katman yükseltme (Q3_K_S)", "",
              "Katman yükseltme: Türkçe profilde en hassas FFN katman(lar)ı daha yüksek türe çıkarılır.", "",
              "| İmatrix | Yöntem | Toplam MB | KLD tr | KLD en | TR değişim |", "| --- | --- | --- | --- | --- | --- |"]
        for im in ("en", "tr"):
            q = f"q3_k_s_{im}"
            ref = tab.get((q, "none"))
            if not ref:
                continue
            items = [(c, v) for (qq, c), v in tab.items() if qq == q]
            items += [(qq.split("+", 1)[1], v) for (qq, c), v in tab.items() if qq.startswith(q + "+up_")]
            for c, v in sorted(items, key=lambda x: x[1]["total"]):
                L.append(f"| {im} | {c} | {v['total']:.1f} | {v.get('tr', float('nan')):.4f} | {v.get('en', float('nan')):.4f} | "
                         f"{(v['tr'] / ref['tr'] - 1) * 100:+.1f}% |")
            L.append("")

    # 3. Eşleştirilmiş bootstrap
    npz = d / "keep_tokens.npz"
    if npz.exists():
        t = np.load(npz)
        L += ["### Eşleştirilmiş karşılaştırmalar (Türkçe token başına KLD farkı, %95 GA)", "",
              "Negatif = ilk seçim daha iyi. GA sıfırı içermiyorsa fark anlamlı.", "",
              "| Taban | Karşılaştırma | Fark | %95 GA |", "| --- | --- | --- | --- |"]
        for q in [q for q in quants if "+up_" not in q]:
            for a, b, desc in PAIRS:
                ka = f"{q}|{a}|tr"
                if b == "rand*":
                    kb = [k for k in t.files if k.startswith(f"{q}|rand") and k.endswith("|tr")]
                    if ka not in t.files or not kb:
                        continue
                    vb = np.mean([t[k] for k in kb], axis=0)
                else:
                    if ka not in t.files or f"{q}|{b}|tr" not in t.files:
                        continue
                    vb = t[f"{q}|{b}|tr"]
                m, lo, hi = boot(t[ka] - vb)
                L.append(f"| {q} | {desc} | {m:+.5f} | {lo:+.5f} … {hi:+.5f} |")
        L.append("")
    return L


def main():
    models = [(p.name, p) for p in sorted(ROOT.iterdir()) if p.is_dir() and (p / "keep.csv").exists()]
    L = ["# P2 pilot 3 raporu: Türkçe nöronları yüksek hassasiyette ayırma", "",
         "Tüm model kuantize; seçilen FFN nöronları (gate/up satırı, down sütunu) F16'da tutulur. "
         "K = Türkçe LAPE nöron sayısı; xM = M*K nöron. err_en = dil bilgisi kullanmayan standart "
         "aktivasyon ağırlıklı hata seçimi; hyb = TR nöronları + kalan bütçe err_en. +MB, nöronları ayrı "
         "tabloda saklamanın ek boyutudur.", ""]
    for tag, d in models:
        L += section(tag, d)
    out = ROOT / "pilot3_report.md"
    out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nRapor: {out}")


if __name__ == "__main__":
    main()
