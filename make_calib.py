#!/usr/bin/env python3
"""Wikipedia'dan İngilizce, Türkçe ve karışık kalibrasyon metni ile
kalibrasyondan ayrık değerlendirme metni üretir.

Kullanım: python3 make_calib.py [kalibrasyon_karakter] [değerlendirme_karakter]
Gereksinim: pip install datasets
Çıktılar: calib/en.txt, calib/tr.txt, calib/mix.txt, calib/eval_en.txt, calib/eval_tr.txt
"""
import random
import sys
from pathlib import Path

from datasets import load_dataset

CHARS = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
EVAL_CHARS = int(sys.argv[2]) if len(sys.argv) > 2 else 80_000
SEED = 2209


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


def sample(lang):
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    rows = iter(ds.shuffle(seed=SEED, buffer_size=10_000))
    # Aynı akıştan önce kalibrasyon, sonra değerlendirme: iki küme ayrık kalır.
    return take(rows, CHARS), take(rows, EVAL_CHARS)


def main():
    out = Path("calib")
    out.mkdir(exist_ok=True)
    (en, eval_en), (tr, eval_tr) = sample("en"), sample("tr")
    (out / "en.txt").write_text("\n\n".join(en))
    (out / "tr.txt").write_text("\n\n".join(tr))
    (out / "eval_en.txt").write_text("\n\n".join(eval_en))
    (out / "eval_tr.txt").write_text("\n\n".join(eval_tr))
    # Karışık set: iki dilden yarı yarıya, aynı toplam uzunlukta.
    mix = en[: len(en) // 2] + tr[: len(tr) // 2]
    random.Random(SEED).shuffle(mix)
    (out / "mix.txt").write_text("\n\n".join(mix))
    for name in ("en", "tr", "mix", "eval_en", "eval_tr"):
        print(name, (out / f"{name}.txt").stat().st_size, "bayt")


if __name__ == "__main__":
    main()
