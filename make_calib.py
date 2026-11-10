#!/usr/bin/env python3
"""Wikipedia'dan dil başına kalibrasyon metni, kalibrasyondan ayrık değerlendirme
metni ve farklı Türkçe paylarında TR+EN karışık kalibrasyon metinleri üretir.

Kullanım: python3 make_calib.py [--langs tr,en,fi,de,ru] [--chars 400000]
                                [--eval-chars 80000] [--shares 0,25,50,75,100]
Gereksinim: pip install datasets
Çıktılar (calib/):
  {dil}.txt        kalibrasyon ve nöron analizi metni
  eval_{dil}.txt   değerlendirme metni (aynı akıştan, kalibrasyondan sonra: ayrık)
  tr{PPP}.txt      Türkçe payı PPP% olan TR+EN karışım (ör. tr025.txt)
  mix.txt          tr050.txt ile aynı (eski betiklerle uyum için)
Var olan dosyalar yeniden üretilmez.
"""
import argparse
import random
from pathlib import Path

from datasets import load_dataset

SEED = 2209
OUT = Path("calib")


def take(rows, chars):
    parts, total = [], 0
    for row in rows:
        text = row["text"].strip()
        if len(text) < 500:
            continue
        parts.append(text[:4000])
        total += len(parts[-1])
        if total >= chars:
            break
    return parts


def sample(lang, chars, eval_chars):
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    rows = iter(ds.shuffle(seed=SEED, buffer_size=10_000))
    # Aynı akıştan önce kalibrasyon, sonra değerlendirme: iki küme ayrık kalır.
    return take(rows, chars), take(rows, eval_chars)


def prefix(parts, chars):
    out, total = [], 0
    for p in parts:
        if total >= chars:
            break
        out.append(p)
        total += len(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="tr,en,fi,de,ru")
    ap.add_argument("--chars", type=int, default=400_000)
    ap.add_argument("--eval-chars", type=int, default=80_000)
    ap.add_argument("--shares", default="0,25,50,75,100")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    calib = {}
    for lang in args.langs.split(","):
        cal, ev = OUT / f"{lang}.txt", OUT / f"eval_{lang}.txt"
        if cal.exists() and ev.exists():
            calib[lang] = cal.read_text().split("\n\n")
            continue
        print(f"== {lang}")
        parts, eval_parts = sample(lang, args.chars, args.eval_chars)
        cal.write_text("\n\n".join(parts))
        ev.write_text("\n\n".join(eval_parts))
        calib[lang] = parts

    # Türkçe payı taraması: aynı toplam uzunlukta TR ve EN belgelerinin karışımı.
    if "tr" in calib and "en" in calib:
        total = min(sum(map(len, calib["tr"])), sum(map(len, calib["en"])))
        for share in map(int, args.shares.split(",")):
            path = OUT / f"tr{share:03d}.txt"
            if path.exists():
                continue
            mix = prefix(calib["tr"], total * share / 100) + prefix(calib["en"], total * (100 - share) / 100)
            random.Random(SEED).shuffle(mix)
            path.write_text("\n\n".join(mix))
        if not (OUT / "mix.txt").exists() and (OUT / "tr050.txt").exists():
            (OUT / "mix.txt").write_text((OUT / "tr050.txt").read_text())

    for path in sorted(OUT.glob("*.txt")):
        print(f"{path.name:14s} {path.stat().st_size:>9d} bayt")


if __name__ == "__main__":
    main()
