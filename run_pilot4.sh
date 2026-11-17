#!/usr/bin/env bash
# Pilot 4: KL kazançları görev doğruluğuna yansıyor mu? TurkishMMLU (sıfır atış), Qwen2.5-1.5B.
# Pilot 3 bittikten sonra çalıştırılır; onun modellerini, imatrix ve nöron dosyalarını kullanır.
# Kollar: F16; Q3_K_S {EN, TR imatrix} × {ayırma yok, standart seçim, Türkçe ağırlıklı seçim};
#         Q4_K_M {EN, TR imatrix} (+ Türkçe ağırlıklı seçim).
# Soru sayısı: N=450 (ders başına 50). Daha hızlı deneme: N=180 ./run_pilot4.sh
# Kaldığı yerden devam eder. Sonuç: results/pilot4_report.md
set -euo pipefail
PY="${PY:-python3}"
N="${N:-450}"
TAG=qwen2.5-1.5b
MODEL=models/qwen2.5-1.5b-instruct-fp16.gguf
HF=models/cfg-qwen2.5-1.5b
WORK="work/$TAG"
K="$WORK/keep"
DATA=data/turkishmmlu
export RES="results/$TAG"
mkdir -p logs "$DATA"

BIN="${BIN:-}"
if [ -z "$BIN" ] && ! command -v llama-quantize >/dev/null; then
  found="$(find ~/llama.cpp ~/llama.cpp-src -name llama-quantize -type f 2>/dev/null | head -1)"
  [ -n "$found" ] && BIN="$(dirname "$found")"
fi
QUANT="${BIN:+$BIN/}llama-quantize"

for f in "$MODEL" "$HF/config.json" "$RES/neurons.npz" "$K/q3_k_s_en.gguf" "$K/q3_k_s_tr.gguf" \
         "$WORK/imatrix-en.dat" "$WORK/imatrix-tr.dat"; do
  [ -f "$f" ] || { echo "Eksik: $f (önce Pilot 2 ve 3)"; exit 1; }
done

# TurkishMMLU test bölümü (herkese açık 900 soru, 9 ders). Veri repoya eklenmez.
for s in Biology Chemistry Geography History Mathematics Philosophy Physics "Religion and Ethics" \
         "Turkish Language and Literature"; do
  f="$DATA/TurkishMMLU_$s.json"
  [ -s "$f" ] || curl -sfL "https://huggingface.co/datasets/AYueksel/TurkishMMLU/resolve/main/test/TurkishMMLU_${s// /%20}.json" -o "$f" \
    || { echo "İndirilemedi: $s"; exit 1; }
done

for lang in en tr; do
  f="$K/q4_k_m_$lang.gguf"
  [ -f "$f" ] || "$QUANT" --imatrix "$WORK/imatrix-$lang.dat" "$MODEL" "$f" Q4_K_M >"logs/p4-q4_$lang.log" 2>&1 \
    || { tail -20 "logs/p4-q4_$lang.log"; echo "llama-quantize hata verdi"; exit 1; }
done

echo "######## Pilot 4 başladı ($(date '+%d.%m %H:%M')), N=$N"
"$PY" analysis/mmlu.py --hf "$HF" --gguf "$MODEL" --data "$DATA" --n "$N" \
  f16=f16:none \
  q3_k_s_en="$K/q3_k_s_en.gguf:none" \
  q3_k_s_en="$K/q3_k_s_en.gguf:err_en_x1" \
  q3_k_s_en="$K/q3_k_s_en.gguf:err_tr_x1" \
  q3_k_s_tr="$K/q3_k_s_tr.gguf:none" \
  q3_k_s_tr="$K/q3_k_s_tr.gguf:err_tr_x1" \
  q4_k_m_en="$K/q4_k_m_en.gguf:none" \
  q4_k_m_tr="$K/q4_k_m_tr.gguf:none" \
  q4_k_m_tr="$K/q4_k_m_tr.gguf:err_tr_x1" 2>&1 | tee logs/p4-mmlu.log
"$PY" analysis/report4.py "$RES" results/pilot4_report.md >/dev/null
echo "Tamamlandı ($(date '+%H:%M')). Rapor: results/pilot4_report.md"
