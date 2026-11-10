#!/usr/bin/env bash
# Aynı F16 modelden her kalibrasyon dili için imatrix hesaplar ve kuantize eder.
#
# Kullanım: ./quantize.sh <model-f16.gguf> [kuantizasyon türleri...]
# Örnek:    ./quantize.sh models/qwen2.5-1.5b-instruct-fp16.gguf Q4_K_M Q2_K
set -euo pipefail

MODEL="$1"; shift
TYPES=("${@:-Q4_K_M Q2_K}")
OUT="models/quant"
mkdir -p "$OUT"
base="$(basename "$MODEL" .gguf)"

for lang in en tr mix; do
  imat="$OUT/imatrix-$lang.dat"
  if [ ! -f "$imat" ]; then
    echo "== imatrix ($lang)"
    llama-imatrix -m "$MODEL" -f "calib/$lang.txt" -o "$imat" --chunks 100
  fi
  for t in ${TYPES[@]}; do
    echo "== $t, kalibrasyon: $lang"
    llama-quantize --imatrix "$imat" "$MODEL" "$OUT/$base-$t-$lang.gguf" "$t"
  done
done

# Kalibrasyonsuz karşılaştırma noktası.
for t in ${TYPES[@]}; do
  llama-quantize "$MODEL" "$OUT/$base-$t-none.gguf" "$t"
done
ls -lh "$OUT"
