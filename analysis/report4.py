#!/usr/bin/env python3
"""Pilot 4 raporu: TurkishMMLU doğruluğu ve F16'ya sadakat, eşleştirilmiş karşılaştırmalar.

Kullanım: python3 analysis/report4.py results/qwen2.5-1.5b [çıktı.md]
"""
import sys
from math import comb
from pathlib import Path

import numpy as np

B = 10000
BASE = "q3_k_s_en|none"   # aracın varsayılanı: İngilizce imatrix, ayırma yok
NAMES = {
    "f16|none": "F16 (referans)",
    "q3_k_s_en|none": "Q3_K_S, EN imatrix (varsayılan)",
    "q3_k_s_tr|none": "Q3_K_S, TR imatrix",
    "q3_k_s_en|err_en_x1": "Q3_K_S, EN imatrix + standart seçim",
    "q3_k_s_en|err_tr_x1": "Q3_K_S, EN imatrix + Türkçe ağırlıklı seçim",
    "q3_k_s_tr|err_tr_x1": "Q3_K_S, TR imatrix + Türkçe ağırlıklı seçim",
    "q3_k_s_tr|err_tr_x4": "Q3_K_S, TR imatrix + Türkçe ağırlıklı seçim (4K)",
    "q4_k_m_en|none": "Q4_K_M, EN imatrix (varsayılan)",
    "q4_k_m_tr|none": "Q4_K_M, TR imatrix",
    "q4_k_m_tr|err_tr_x1": "Q4_K_M, TR imatrix + Türkçe ağırlıklı seçim",
}


def boot(diff, rng):
    idx = rng.integers(0, len(diff), (B, len(diff)))
    m = diff[idx].mean(1)
    return diff.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def mcnemar(a, b):
    """Kesin iki yönlü McNemar: a doğru/b yanlış ve tersi."""
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, p), n01, n10


def main():
    res_dir = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/pilot4_report.md")
    d = dict(np.load(res_dir / "mmlu.npz", allow_pickle=True))
    gold, subj = d.pop("gold"), d.pop("subject")
    arms = [k for k in NAMES if k in d] + [k for k in d if k not in NAMES]
    rng = np.random.default_rng(0)
    n = len(gold)

    def probs(k):
        m = d[k].astype(np.float64)
        return np.exp(m - np.logaddexp.reduce(m, axis=1, keepdims=True))

    corr = {k: d[k].argmax(1) == gold for k in arms}
    lpc = {k: np.log(probs(k)[np.arange(n), gold]) for k in arms}
    f16 = "f16|none" if "f16|none" in d else None

    L = [f"# P2 pilot 4 raporu: TurkishMMLU (sıfır atış, {n} soru, {len(set(subj))} ders)", "",
         "Model: Qwen2.5-1.5B-Instruct. Nöron ayırma Pilot 3'teki benzetimle yapılır (seçilen FFN nöronları F16).",
         "Şans düzeyi yaklaşık %20 (5 seçenek). 'F16 ile aynı' ve 'KL (F16)' kuantizasyonun F16 modelden ne kadar",
         "saptığını gösterir ve doğruluktan daha duyarlıdır.", "",
         "| Kol | Doğruluk | %95 GA | log P(doğru) | F16 ile aynı cevap | KL (F16, seçenekler) |",
         "| --- | --- | --- | --- | --- | --- |"]
    for k in arms:
        a, lo, hi = boot(corr[k].astype(float), rng)
        same = kl = "–"
        if f16 and k != f16:
            same = f"{(d[k].argmax(1) == d[f16].argmax(1)).mean():.3f}"
            p0, p1 = probs(f16), probs(k)
            kl = f"{np.nansum(np.where(p0 > 0, p0 * (np.log(p0) - np.log(np.clip(p1, 1e-12, 1))), 0), axis=1).mean():.4f}"
        L.append(f"| {NAMES.get(k, k)} | {a:.3f} | {lo:.3f} … {hi:.3f} | {lpc[k].mean():.3f} | {same} | {kl} |")

    L += ["", f"## Eşleştirilmiş karşılaştırmalar (taban: {NAMES.get(BASE, BASE)})", "",
          "Doğruluk farkı ve log P(doğru) farkı: pozitif = karşılaştırılan kol daha iyi. GA sıfırı içermiyorsa fark anlamlı.", "",
          "| Kol | Doğruluk farkı | %95 GA | McNemar p (kazanan/kaybeden soru) | log P(doğru) farkı | %95 GA |",
          "| --- | --- | --- | --- | --- | --- |"]
    pairs = [(BASE, k) for k in arms if k not in (BASE, f16) and BASE in d]
    if "q4_k_m_en|none" in d:
        pairs += [("q4_k_m_en|none", k) for k in arms if k.startswith("q4_k_m") and k != "q4_k_m_en|none"]
    for b, k in pairs:
        da, alo, ahi = boot(corr[k].astype(float) - corr[b].astype(float), rng)
        p, n01, n10 = mcnemar(corr[k], corr[b])
        dl, llo, lhi = boot(lpc[k] - lpc[b], rng)
        L.append(f"| {NAMES.get(k, k)} vs {NAMES.get(b, b)} | {da:+.3f} | {alo:+.3f} … {ahi:+.3f} | "
                 f"{p:.3f} ({n01}/{n10}) | {dl:+.4f} | {llo:+.4f} … {lhi:+.4f} |")

    if f16:
        L += ["", "## Kayıp ne kadar geri kazanıldı", "",
              "F16 ile varsayılan arasındaki log P(doğru) farkının yüzde kaçı kapatıldı.", ""]
        gap = lpc[f16].mean() - lpc[BASE].mean() if BASE in d else None
        for k in arms:
            if gap and k not in (f16, BASE) and k.startswith("q3"):
                L.append(f"- {NAMES.get(k, k)}: %{100 * (lpc[k].mean() - lpc[BASE].mean()) / gap:.0f}")

    L += ["", "## Ders bazında doğruluk", "", "| Ders | " + " | ".join(NAMES.get(k, k) for k in arms) + " |",
          "| --- |" + " --- |" * len(arms)]
    for s in sorted(set(subj)):
        m = subj == s
        L.append(f"| {s} | " + " | ".join(f"{corr[k][m].mean():.2f}" for k in arms) + " |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
