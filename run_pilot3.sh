#!/usr/bin/env bash
# Pilot 3 (tahmini 3–4 saat): Türkçe nöronları yüksek hassasiyette ayırma (Y1').
# Pilot 2 bittikten sonra çalıştırılır; onun modellerini, imatrix ve nöron dosyalarını kullanır.
#   Qwen2.5-1.5B: tam kapsam (Q3_K_S ve Q2_K, imatrix en/tr, katman yükseltme rakipleri)
#   Qwen2.5-3B:   çekirdek kollar (Q3_K_S, imatrix en/tr)
# Kaldığı yerden devam eder. Sonuç: results/pilot3_report.md
set -uo pipefail
export BIN="${BIN:-}" PY="${PY:-python3}"
mkdir -p logs
FAILED=()

run() {
  local name="$1"; shift
  echo "######## $name başladı ($(date '+%d.%m %H:%M'))"
  if "$@" 2>&1 | tee "logs/$name.log"; [ "${PIPESTATUS[0]}" -eq 0 ]; then
    echo "######## $name bitti ($(date '+%H:%M'))"
  else
    echo "######## $name HATA verdi, günlük: logs/$name.log"
    FAILED+=("$name")
  fi
}

if [ -z "$BIN" ] && ! command -v llama-quantize >/dev/null; then
  found="$(find ~/llama.cpp ~/llama.cpp-src -name llama-quantize -type f 2>/dev/null | head -1)"
  [ -n "$found" ] && BIN="$(dirname "$found")"
fi
[ -n "$BIN" ] || command -v llama-quantize >/dev/null || { echo "llama-quantize bulunamadı: BIN=/klasör ./run_pilot3.sh"; exit 1; }
export BIN
for d in qwen2.5-1.5b qwen2.5-3b; do
  [ -f "results/$d/neurons.npz" ] || { echo "Eksik: results/$d/neurons.npz (önce ./run_all.sh)"; exit 1; }
done

run p3-qwen2.5-1.5b env TAG=qwen2.5-1.5b MODEL=models/qwen2.5-1.5b-instruct-fp16.gguf HF=models/cfg-qwen2.5-1.5b \
  OUTLIER_SCOPE=full ./pilot.sh outlier
run p3-qwen2.5-3b env TAG=qwen2.5-3b MODEL=models/qwen2.5-3b-instruct-fp16-00001-of-00002.gguf HF=models/cfg-qwen2.5-3b \
  OUTLIER_SCOPE=core ./pilot.sh outlier
run p3-rapor "$PY" analysis/report3.py results

echo
if [ ${#FAILED[@]} -eq 0 ]; then echo "Tamamlandı. Rapor: results/pilot3_report.md"
else echo "Hata veren bölümler: ${FAILED[*]} (logs/ altına bak; tekrar çalıştırınca kaldığı yerden devam eder)"; fi
