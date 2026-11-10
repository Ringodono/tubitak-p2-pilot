#!/usr/bin/env bash
# Pilot 2'nin tamamı, tek komutla (tahmini 5–7 saat, ~35 GB disk):
#   1. modelleri ve metinleri indirir/hazırlar
#   2. Qwen2.5-1.5B: tüm aşamalar (olgu, mekanizma M1–M3, yöntem Y1–Y3)
#   3. EuroLLM-1.7B (başka aile, Türkçe ve Fince destekli): olgu ve kalibrasyon
#   4. Qwen2.5-3B (büyüklük): olgu, kalibrasyon ve nöron analizi
#   5. results/pilot2_report.md
# Bir model hata verirse günlüğe yazılır ve sıradakine geçilir; tekrar çalıştırınca
# her aşama kaldığı yerden devam eder.
#
# Kullanım: BIN=~/llama.cpp/bin ./run_all.sh      (BIN: llama.cpp ikilileri)
set -uo pipefail
export BIN="${BIN:-}" PY="${PY:-python3}"
mkdir -p logs models
FAILED=()

run() { # run AD komut...
  local name="$1"; shift
  echo "######## $name başladı ($(date '+%d.%m %H:%M'))"
  if "$@" 2>&1 | tee "logs/$name.log"; [ "${PIPESTATUS[0]}" -eq 0 ]; then
    echo "######## $name bitti ($(date '+%H:%M'))"
  else
    echo "######## $name HATA verdi, günlük: logs/$name.log"
    FAILED+=("$name")
  fi
}

download() {
  local f
  dl() { [ -f "models/$2" ] || hf download "$1" "$2" --local-dir models; }
  dl Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-fp16.gguf
  dl mradermacher/EuroLLM-1.7B-Instruct-GGUF EuroLLM-1.7B-Instruct.f16.gguf
  for f in qwen2.5-3b-instruct-fp16-00001-of-00002.gguf qwen2.5-3b-instruct-fp16-00002-of-00002.gguf; do
    dl Qwen/Qwen2.5-3B-Instruct-GGUF "$f"
  done
  # Python analizleri için yalnız yapılandırma ve tokenizer (ağırlıklar GGUF'tan okunur)
  for m in 1.5B 3B; do
    [ -f "models/cfg-qwen2.5-${m,,}/config.json" ] || hf download "Qwen/Qwen2.5-$m-Instruct" \
      config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt \
      --local-dir "models/cfg-qwen2.5-${m,,}"
  done
}

# Gerekli Python paketleri: eksikse hemen dur (her model aynı yerde takılmasın)
"$PY" -c "import datasets, gguf, numpy, torch, transformers, safetensors" 2>/dev/null || {
  echo "Eksik Python paketi. Pilotu çalıştırdığın ortamda (ör. .venv) şunu çalıştır:"
  echo "  pip install datasets gguf numpy transformers safetensors huggingface_hub"
  echo "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
  exit 1
}

run indirme download
run metinler "$PY" make_calib.py
[ -f calib/eval_fi.txt ] && [ -f calib/tr050.txt ] || { echo "Metinler hazırlanamadı, günlük: logs/metinler.log"; exit 1; }

run qwen2.5-1.5b env TAG=qwen2.5-1.5b MODEL=models/qwen2.5-1.5b-instruct-fp16.gguf HF=models/cfg-qwen2.5-1.5b \
  ./pilot.sh check base profile calib share alloc keep neurons qerror repair fisher boost

run eurollm-1.7b env TAG=eurollm-1.7b MODEL=models/EuroLLM-1.7B-Instruct.f16.gguf \
  ./pilot.sh check base profile calib share

run qwen2.5-3b env TAG=qwen2.5-3b MODEL=models/qwen2.5-3b-instruct-fp16-00001-of-00002.gguf HF=models/cfg-qwen2.5-3b \
  ./pilot.sh check base profile calib neurons qerror

run rapor "$PY" analysis/report2.py results

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "Tamamlandı. Rapor: results/pilot2_report.md"
else
  echo "Hata veren bölümler: ${FAILED[*]} (logs/ altındaki günlüklere bak; tekrar çalıştırınca kaldığı yerden devam eder)"
fi
