#!/usr/bin/env python3
"""Pilot 2 raporu: results/<model>/ altındaki tüm çıktıları tek markdown raporda toplar.

Kullanım: python3 analysis/report2.py [results]
Çıktı: results/pilot2_report.md
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "results")


def rows(path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def ranks(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):  # eşitlikler için ortalama sıra
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def fmt(x, d=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def profile(d):
    """{(katman, grup): {dil: kld - taban}}"""
    prof, floor = {}, {}
    for r in rows(d / "profile.csv"):
        if r["layer"] == "-1":
            floor[r["lang"]] = float(r["kld"])
        else:
            prof.setdefault((int(r["layer"]), r["group"]), {})[r["lang"]] = float(r["kld"])
    return {k: {l: v - floor.get(l, 0.0) for l, v in x.items()} for k, x in prof.items()}, floor


def table(d, name):
    """{model: {dil: kld}} ve boyutlar"""
    out, size = {}, {}
    for r in rows(d / name):
        out.setdefault(r["model"], {})[r["lang"]] = float(r["kld"])
        size[r["model"]] = r["size_mb"]
    return out, size


def section_phenomenon(models):
    L = ["## 1. Olgu: Türkçe kırılganlığı ve katman profili", "",
         "| Model | TR/EN KLD (Q3_K_S, imatrix yok) | FI/EN | Profil ρ (TR–EN) | TR/EN ilk %20 katman | orta | son %20 |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for tag, d in models:
        prof, _ = profile(d)
        cal, _ = table(d, "calib.csv")
        base = cal.get("q3_k_s+imatrix_none", {})
        tr_en = base["tr"] / base["en"] if "tr" in base and "en" in base else float("nan")
        fi_en = base["fi"] / base["en"] if "fi" in base and "en" in base else float("nan")
        keys = sorted(k for k in prof if "tr" in prof[k] and "en" in prof[k])
        rho = spearman([prof[k]["tr"] for k in keys], [prof[k]["en"] for k in keys]) if keys else float("nan")
        bands = ["-"] * 3
        if keys:
            n = max(k[0] for k in keys) + 1
            per_layer = {}
            for (layer, _g) in keys:
                per_layer.setdefault(layer, [0.0, 0.0])
                per_layer[layer][0] += prof[(layer, _g)]["tr"]
                per_layer[layer][1] += prof[(layer, _g)]["en"]
            ratio = np.array([per_layer[i][0] / max(per_layer[i][1], 1e-12) for i in range(n)])
            e = max(1, round(n * 0.2))
            bands = [fmt(ratio[:e].mean(), 2), fmt(ratio[e:n - e].mean(), 2), fmt(ratio[n - e:].mean(), 2)]
        L.append(f"| {tag} | {fmt(tr_en, 2)} | {fmt(fi_en, 2)} | {fmt(rho)} | {bands[0]} | {bands[1]} | {bands[2]} |")
    return L + [""]


def section_neurons(tag, d):
    L = []
    nl = rows(d / "neurons_layers.csv")
    if nl:
        prof, _ = profile(d)
        langs = [c[2:] for c in nl[0] if c.startswith("n_") and c != "n_specific"]
        tot = {l: sum(int(r[f"n_{l}"]) for r in nl) for l in langs}
        L += [f"### {tag}: dile özgü nöronlar (LAPE)", "", "Toplam: " + ", ".join(f"{l} {n}" for l, n in tot.items()), ""]
        if prof:
            layers = [int(r["layer"]) for r in nl]
            ratio = []
            for i in layers:
                trs = sum(v["tr"] for (l, _g), v in prof.items() if l == i and "tr" in v)
                ens = sum(v["en"] for (l, _g), v in prof.items() if l == i and "en" in v)
                ratio.append(trs / max(ens, 1e-12))
            rho = spearman([int(r["n_tr"]) for r in nl], ratio)
            L += [f"Katman başına TR nöron sayısı ile TR/EN KLD oranı arasında Spearman ρ = **{fmt(rho)}** "
                  "(pozitifse U şekli Türkçe nöronların yerleşimiyle örtüşüyor).", ""]
    qe = rows(d / "qerror.csv")
    if qe:
        L += [f"### {tag}: kuantizasyon hatasının nöronlara dağılımı", "",
              "Enerji payı: Türkçe çıktı hata enerjisinin bu kümeden gelen kısmı (katman ortalaması). "
              "Yoğunlaşma = enerji payı / nöron payı (1 = eşit dağılım).", "",
              "| Model | Küme | Nöron | Göreli hata | TR enerji payı | TR yoğunlaşma | EN enerji payı |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        agg = {}
        for r in qe:
            agg.setdefault((r["model"], r["set"]), []).append(r)
        n_all = {m: sum(int(r["n"]) for r in rs) for (m, s), rs in agg.items() if s == "all"}
        for (m, s), rs in sorted(agg.items()):
            if s == "all":
                continue
            n = sum(int(r["n"]) for r in rs)
            rel = sum(float(r["rel_err"]) * int(r["n"]) for r in rs) / n
            n_layers = len([r for r in agg[(m, "all")]])
            etr = sum(float(r["energy_share_tr"]) for r in rs) / n_layers
            een = sum(float(r.get("energy_share_en", "nan")) for r in rs) / n_layers
            conc = etr / (n / n_all[m]) if n_all.get(m) else float("nan")
            L.append(f"| {m} | {s} | {n} | {fmt(rel, 4)} | {fmt(etr, 4)} | {fmt(conc, 1)} | {fmt(een, 4)} |")
        L.append("")
    ic = rows(d / "imatrix_compare.csv")
    if ic:
        L += [f"### {tag}: imatrix önemi, Türkçe / İngilizce kalibrasyon (medyan oran)", "",
              "| Nöron kümesi | n | TR imatrix / EN imatrix |", "| --- | --- | --- |"]
        L += [f"| {r['küme']} | {r['n']} | {r['medyan_A/B']} |" for r in ic] + [""]
    return L


def bootstrap_diff(a, b, n=2000, seed=0):
    """Eşleştirilmiş token farkı ortalaması (a - b) ve %95 GA."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def section_repair(tag, d):
    rp = rows(d / "repair.csv")
    if not rp:
        return []
    kl = {}
    for r in rp:
        kl.setdefault(r["config"], {})[r["lang"]] = (float(r["kld"]), int(r["n_restored"]))
    langs = sorted({r["lang"] for r in rp}, key=lambda x: ["tr", "en", "fi"].index(x) if x in ("tr", "en", "fi") else 9)
    L = [f"### {tag}: nedensel test, yalnız FFN kuantize (Q3_K_S, imatrix EN), nöron kümesi onarılıyor", "",
         "| Yapılandırma | Onarılan nöron | " + " | ".join(f"KLD {l} (değişim)" for l in langs) + " |",
         "| --- | --- | " + " | ".join("---" for _ in langs) + " |"]
    q = kl.get("quant", {})
    for c, v in kl.items():
        cells = []
        for l in langs:
            if l in v:
                ch = (v[l][0] / q[l][0] - 1) * 100 if l in q and c != "quant" else None
                cells.append(f"{v[l][0]:.4f}" + (f" ({ch:+.1f}%)" if ch is not None else ""))
            else:
                cells.append("-")
        L.append(f"| {c} | {next(iter(v.values()))[1]} | " + " | ".join(cells) + " |")
    npz = d / "repair_tokens.npz"
    if npz.exists():
        t = np.load(npz)
        rand = [k for k in t.files if k.startswith("restore_rand") and k.endswith("|tr")]
        if "restore_tr|tr" in t.files and rand:
            r_mean = np.mean([t[k] for k in rand], axis=0)
            m, lo, hi = bootstrap_diff(t["restore_tr|tr"], r_mean)
            L += ["", f"TR onarımı − rastgele onarım (Türkçe token başına KLD farkı): **{m:.5f}** "
                  f"(%95 GA {lo:.5f} … {hi:.5f}); GA sıfırın altındaysa TR nöronları rastgeleden anlamlı fazla onarıyor."]
        L += ["", f"### {tag}: token türüne göre KLD (M3, yalnız FFN kuantize)", "",
              "| Dil | Kelime başı | Kelime içi devam | Diğer | Devam / baş |", "| --- | --- | --- | --- | --- |"]
        for l in langs:
            if f"class|{l}" in t.files and f"quant|{l}" in t.files:
                c, k = t[f"class|{l}"], t[f"quant|{l}"]
                means = [float(k[c == i].mean()) if (c == i).any() else float("nan") for i in range(3)]
                L.append(f"| {l} | {fmt(means[0], 4)} | {fmt(means[1], 4)} | {fmt(means[2], 4)} | {fmt(means[1] / means[0], 2)} |")
    return L + [""]


def section_csv(tag, d, name, title, note, ref=None):
    tab, size = table(d, name)
    if not tab:
        return []
    langs = [l for l in ("tr", "en", "fi") if any(l in v for v in tab.values())]
    L = [f"### {tag}: {title}", "", note, "",
         "| Model | MB | " + " | ".join(f"KLD {l}" for l in langs) + (" | TR değişim (ref.) |" if ref else " |"),
         "| --- | --- | " + " | ".join("---" for _ in langs) + (" | --- |" if ref else " |")]
    for m, v in tab.items():
        cells = [fmt(v.get(l), 4) for l in langs]
        extra = ""
        if ref:
            r = ref(m)
            if r and r in tab and "tr" in tab[r] and "tr" in v:
                extra = f" | {(v['tr'] / tab[r]['tr'] - 1) * 100:+.1f}%"
            else:
                extra = " | "
        L.append(f"| {m} | {size[m]} | " + " | ".join(cells) + extra + " |")
    return L + [""]


def section_fisher(tag, d):
    fi = rows(d / "fisher.csv")
    if not fi:
        return []
    prof, _ = profile(d)
    est = {}
    for r in fi:
        est.setdefault((int(r["layer"]), r["group"]), {})[r["lang"]] = float(r["est"])
    L = [f"### {tag}: Fisher kestirimi ile ölçülen hassasiyet (Y2)", "", "| Dil | Spearman ρ (kestirim vs ölçüm) |", "| --- | --- |"]
    for lang in sorted({r["lang"] for r in fi}):
        keys = [k for k in est if lang in est[k] and k in prof and lang in prof[k]]
        L.append(f"| {lang} | {fmt(spearman([est[k][lang] for k in keys], [prof[k][lang] for k in keys]))} |")
    keys = [k for k in est if all(l in est[k] for l in ("tr", "en")) and k in prof and all(l in prof[k] for l in ("tr", "en"))]
    if keys:
        r = spearman([est[k]["tr"] / max(est[k]["en"], 1e-12) for k in keys], [prof[k]["tr"] / max(prof[k]["en"], 1e-12) for k in keys])
        L += ["", f"TR/EN oranının kestirimi ile ölçümü arasında ρ = **{fmt(r)}** (dile özgü hassasiyet farkını da yakalıyor mu?)."]
    return L + [""]


def main():
    models = [(p.name, p) for p in sorted(ROOT.iterdir()) if p.is_dir() and (p / "profile.csv").exists()]
    if (ROOT / "profile.csv").exists():
        models.insert(0, ("v1 (results/)", ROOT))
    L = ["# P2 pilot 2 raporu", "", "Modeller: " + ", ".join(t for t, _ in models), ""]
    L += section_phenomenon(models)
    L += ["## 2. Mekanizma (M1–M3)", ""]
    for tag, d in models:
        L += section_neurons(tag, d) + section_repair(tag, d)
    L += ["## 3. Yöntem ön denemeleri (Y1–Y3)", ""]
    for tag, d in models:
        L += section_csv(tag, d, "boost.csv", "Türkçe nöron ağırlıklı imatrix (Y1)",
                         "Referans: aynı türde imatrix_en. rand = TR kadar rastgele nöron; en/fi = başka dilin nöronları (kontrol).",
                         ref=lambda m: m.split("+")[0] + "+imatrix_en")
        L += section_fisher(tag, d)
        L += section_csv(tag, d, "share.csv", "Kalibrasyonda Türkçe payı (Y3)", "trPPP = imatrix metninde %PPP Türkçe, gerisi İngilizce.",
                         ref=lambda m: m.split("+")[0] + "+tr000")
        L += section_csv(tag, d, "calib.csv", "Kalibrasyon dili", "Q3_K_S.", ref=lambda m: "q3_k_s+imatrix_en")
        L += section_csv(tag, d, "alloc.csv", "Dile göre bit dağıtımı (eşit boyut)", "Taban Q3_K_S, seçilen 8 grup Q5_K.",
                         ref=lambda m: "q3_k_s+en_aware+imatrix_en")
    out = ROOT / "pilot2_report.md"
    out.write_text("\n".join(L))
    print("\n".join(L[:40]))
    print(f"\nTam rapor: {out}")


if __name__ == "__main__":
    main()
