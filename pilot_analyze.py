#!/usr/bin/env python3
"""Pilot sonuçlarını ayrıştırır, analiz eder ve rapor üretir.

Alt komutlar:
  parse LOG                  llama-perplexity KL çıktısından "kld hata top1" yazar
  select LANG K              LANG için en hassas K dikkat + K ileri besleme grubunu
                             --tensor-type desenleri olarak yazar (her satıra bir)
  report                     results/*.csv dosyalarından results/pilot_results.md üretir
"""
import csv
import re
import sys
from pathlib import Path

RES = Path("results")


def parse(log):
    text = Path(log).read_text(errors="replace")
    kld = re.search(r"Mean\s+KLD:\s*([\d.eE+-]+)\s*±\s*([\d.eE+-]+)", text)
    top = re.search(r"Same top p:\s*([\d.]+)", text)
    if not kld:
        sys.exit(f"KLD satırı bulunamadı: {log}")
    print(kld.group(1), kld.group(2), top.group(1) if top else "nan")


def load(name):
    path = RES / name
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def ranks(values):
    # Eşitliklerde ortalama sıra.
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb)


def profile():
    """{(katman, grup): {dil: kld}} ve Q8_0 tabanı."""
    rows = load("profile.csv")
    prof, floor = {}, {}
    for r in rows:
        if r["layer"] == "-1":
            floor[r["lang"]] = float(r["kld"])
            continue
        prof.setdefault((int(r["layer"]), r["group"]), {})[r["lang"]] = float(r["kld"])
    prof = {k: v for k, v in prof.items() if "tr" in v and "en" in v}
    return prof, floor


def select(lang, k):
    prof, floor = profile()
    by_group = {}
    for (layer, group), v in prof.items():
        by_group.setdefault(group, []).append((v[lang] - floor.get(lang, 0.0), layer))
    for group, items in sorted(by_group.items()):
        for _, layer in sorted(items, reverse=True)[:k]:
            print(f"blk\\.{layer}\\.{group}_.*")


def report():
    prof, floor = profile()
    lines = ["# P2 pilot sonuçları", ""]
    if prof:
        keys = sorted(prof)
        tr = [prof[k]["tr"] - floor.get("tr", 0.0) for k in keys]
        en = [prof[k]["en"] - floor.get("en", 0.0) for k in keys]
        rho = spearman(tr, en)
        n_top = max(1, round(len(keys) * 0.2))
        rt, re_ = ranks([-x for x in tr]), ranks([-x for x in en])
        top_tr = {keys[i] for i in range(len(keys)) if rt[i] <= n_top}
        top_en = {keys[i] for i in range(len(keys)) if re_[i] <= n_top}
        overlap = len(top_tr & top_en) / n_top
        lines += [
            "## H1: Türkçe ve İngilizce hassasiyet sıralaması",
            "",
            f"- Grup sayısı: {len(keys)} (katman × {{dikkat, ileri besleme}})",
            f"- Q8_0 tabanı (KLD): TR {floor.get('tr', float('nan')):.5f}, EN {floor.get('en', float('nan')):.5f}",
            f"- Spearman ρ (TR vs EN): **{rho:.3f}**",
            f"- En hassas %20'lik grupların örtüşmesi: **{overlap:.0%}** ({len(top_tr & top_en)}/{n_top})",
            f"- H1 eşiği (ρ < 0,8 veya örtüşme < %70): **{'sağlandı' if rho < 0.8 or overlap < 0.7 else 'sağlanmadı'}**",
            "",
            "Sıra farkı en büyük gruplar (pozitif = Türkçede daha hassas):",
            "",
            "| Katman | Grup | KLD TR | KLD EN | Sıra TR | Sıra EN | Fark |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        diffs = sorted(range(len(keys)), key=lambda i: re_[i] - rt[i], reverse=True)
        for i in diffs[:5] + diffs[-5:]:
            layer, group = keys[i]
            lines.append(
                f"| {layer} | {group} | {tr[i]:.5f} | {en[i]:.5f} | {rt[i]:.0f} | {re_[i]:.0f} | {re_[i] - rt[i]:+.0f} |"
            )
        lines += ["", "Tüm gruplar:", "", "| Katman | Grup | KLD TR | KLD EN | TR/EN |", "| --- | --- | --- | --- | --- |"]
        for i, (layer, group) in enumerate(keys):
            ratio = tr[i] / en[i] if en[i] > 0 else float("nan")
            lines.append(f"| {layer} | {group} | {tr[i]:.5f} | {en[i]:.5f} | {ratio:.2f} |")
        lines.append("")

    for name, title, note in (
        ("alloc.csv", "H2 ön denemesi: dile göre bit dağıtımı", "Taban Q3_K_S; seçilen gruplar Q5_K. TR ve EN kolları aynı sayıda dikkat ve ileri besleme grubu yükseltir, boyutları eşittir."),
        ("calib.csv", "Kalibrasyon dili ön denemesi", "Q3_K_S; imatrix metni değişiyor."),
    ):
        rows = load(name)
        if not rows:
            continue
        lines += [f"## {title}", "", note, "", "| Model | Boyut (MB) | KLD TR | KLD EN | Top-1 aynı TR (%) | Top-1 aynı EN (%) |", "| --- | --- | --- | --- | --- | --- |"]
        by_model = {}
        for r in rows:
            by_model.setdefault(r["model"], {"size": r["size_mb"]})[r["lang"]] = r
        for model, v in by_model.items():
            t, e = v.get("tr", {}), v.get("en", {})
            lines.append(
                f"| {model} | {v['size']} | {t.get('kld', '-')} | {e.get('kld', '-')} | {t.get('top1', '-')} | {e.get('top1', '-')} |"
            )
        lines.append("")

    (RES / "pilot_results.md").write_text("\n".join(lines))
    print("\n".join(lines[:16]))
    print(f"\nTam rapor: {RES / 'pilot_results.md'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "parse":
        parse(sys.argv[2])
    elif cmd == "select":
        select(sys.argv[2], int(sys.argv[3]))
    elif cmd == "report":
        report()
    else:
        sys.exit(__doc__)
