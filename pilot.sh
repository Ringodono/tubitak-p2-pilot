#!/usr/bin/env bash
# P2 pilotu: katman düzeyinde Türkçe/İngilizce kuantizasyon hassasiyeti,
# dile göre bit dağıtımı ön denemesi ve kalibrasyon dili ön denemesi.
#
# Kullanım: ./pilot.sh [aşama...]
#   Aşamalar: check base profile alloc calib report   (varsayılan: hepsi, sırayla)
# Ortam değişkenleri:
#   MODEL   F16 GGUF yolu (varsayılan: models/qwen2.5-1.5b-instruct-fp16.gguf)
#   BIN     llama.cpp ikili dizini (varsayılan: PATH'teki llama-* komutları)
#   NGL     ekran kartına alınacak katman sayısı (varsayılan: 99; yalnız CPU için 0)
#   CHUNKS  değerlendirme parça sayısı, 512 token (varsayılan: 20)
#   LOW     profilde bir gruba verilen düşük tür (varsayılan: q2_K)
#   K       dağıtım denemesinde dil başına yükseltilen grup sayısı, tür başına (varsayılan: 4)
# Her aşama kaldığı yerden devam eder; sonuçlar results/ altına yazılır.
set -euo pipefail

MODEL="${MODEL:-models/qwen2.5-1.5b-instruct-fp16.gguf}"
BIN="${BIN:-}"
NGL="${NGL:-99}"
CHUNKS="${CHUNKS:-20}"
LOW="${LOW:-q2_K}"
K="${K:-4}"
CTX=512
WORK=work
RES=results
mkdir -p "$WORK" "$RES"

bin() { if [ -n "$BIN" ]; then echo "$BIN/$1"; else echo "$1"; fi; }
QUANT="$(bin llama-quantize)"
PPL="$(bin llama-perplexity)"
IMAT="$(bin llama-imatrix)"

size_mb() { echo $(( $(stat -c %s "$1" 2>/dev/null || stat -f %z "$1") / 1048576 )); }

# kld MODEL LANG LOG -> "kld hata top1"
kld() {
  "$PPL" -m "$1" -c "$CTX" --chunks "$CHUNKS" -ngl "$NGL" \
    --kl-divergence-base "$WORK/base_$2.kld" --kl-divergence >"$3" 2>&1 \
    || { echo "llama-perplexity hata verdi, günlük: $3" >&2; tail -20 "$3" >&2; exit 1; }
  python3 pilot_analyze.py parse "$3"
}

stage_check() {
  for f in "$MODEL" calib/eval_tr.txt calib/eval_en.txt calib/tr.txt calib/en.txt calib/mix.txt; do
    [ -f "$f" ] || { echo "Eksik: $f (README.md adım 2–3)" >&2; exit 1; }
  done
  for c in "$QUANT" "$PPL" "$IMAT"; do
    command -v "$c" >/dev/null || { echo "Bulunamadı: $c (BIN değişkenini ayarla)" >&2; exit 1; }
  done
  N_LAYERS="$(python3 -c "
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
f = next(f for k, f in r.fields.items() if k.endswith('.block_count'))
print(int(f.parts[f.data[0]][0]))" "$MODEL")"
  echo "$N_LAYERS" >"$WORK/n_layers"
  echo "Model: $MODEL, katman: $N_LAYERS, NGL=$NGL, CHUNKS=$CHUNKS"
}

stage_base() {
  for lang in tr en; do
    if [ ! -f "$WORK/base_$lang.kld" ]; then
      echo "== F16 taban logitleri ($lang)"
      "$PPL" -m "$MODEL" -f "calib/eval_$lang.txt" -c "$CTX" --chunks "$CHUNKS" -ngl "$NGL" \
        --kl-divergence-base "$WORK/base_$lang.kld" >"$WORK/base_$lang.log" 2>&1 \
        || { tail -20 "$WORK/base_$lang.log" >&2; exit 1; }
    fi
  done
}

stage_profile() {
  local n_layers out="$RES/profile.csv"
  n_layers="$(cat "$WORK/n_layers")"
  [ -f "$out" ] || echo "layer,group,lang,kld,kld_err,top1" >"$out"
  # Q8_0 tabanı: hiçbir grup düşürülmeden.
  if ! grep -q '^-1,' "$out"; then
    echo "== Q8_0 tabanı"
    "$QUANT" "$MODEL" "$WORK/tmp.gguf" Q8_0 >"$WORK/q.log" 2>&1
    for lang in tr en; do
      read -r v e t < <(kld "$WORK/tmp.gguf" "$lang" "$WORK/kld.log")
      echo "-1,none,$lang,$v,$e,$t" >>"$out"
    done
  fi
  for ((i = 0; i < n_layers; i++)); do
    for g in attn ffn; do
      grep -q "^$i,$g,en," "$out" && continue
      echo "== katman $i / $g -> $LOW"
      "$QUANT" --tensor-type "blk\.$i\.${g}_.*=$LOW" "$MODEL" "$WORK/tmp.gguf" Q8_0 >"$WORK/q.log" 2>&1 \
        || { tail -20 "$WORK/q.log" >&2; exit 1; }
      for lang in tr en; do
        read -r v e t < <(kld "$WORK/tmp.gguf" "$lang" "$WORK/kld.log")
        echo "$i,$g,$lang,$v,$e,$t" >>"$out"
      done
    done
  done
  rm -f "$WORK/tmp.gguf"
}

# build_eval NAME OUT.csv [llama-quantize seçenekleri...]
build_eval() {
  local name="$1" out="$2"; shift 2
  grep -q "^$name," "$out" 2>/dev/null && return
  echo "== $name"
  "$QUANT" "$@" "$MODEL" "$WORK/$name.gguf" Q3_K_S >"$WORK/q.log" 2>&1 || { tail -20 "$WORK/q.log" >&2; exit 1; }
  local mb; mb="$(size_mb "$WORK/$name.gguf")"
  for lang in tr en; do
    read -r v e t < <(kld "$WORK/$name.gguf" "$lang" "$WORK/kld.log")
    echo "$name,$mb,$lang,$v,$e,$t" >>"$out"
  done
  rm -f "$WORK/$name.gguf"
}

stage_alloc() {
  local out="$RES/alloc.csv"
  [ -f "$out" ] || echo "model,size_mb,lang,kld,kld_err,top1" >"$out"
  build_eval q3_k_s "$out"
  # sel: seçim ölçütü, im: imatrix dili (boş = imatrix yok)
  for arm in en: tr: tr_excess: en:en tr:tr tr_excess:tr; do
    local sel="${arm%%:*}" im="${arm#*:}" args=() name
    while read -r pat; do args+=(--tensor-type "$pat=q5_K"); done < <(python3 pilot_analyze.py select "$sel" "$K")
    name="q3_k_s+${sel}_aware"
    if [ -n "$im" ]; then
      [ -f "$WORK/imatrix-$im.dat" ] || { echo "Önce calib aşaması: $WORK/imatrix-$im.dat yok" >&2; exit 1; }
      args+=(--imatrix "$WORK/imatrix-$im.dat")
      name="$name+imatrix_$im"
    fi
    build_eval "$name" "$out" "${args[@]}"
  done
}

stage_calib() {
  local out="$RES/calib.csv"
  [ -f "$out" ] || echo "model,size_mb,lang,kld,kld_err,top1" >"$out"
  for lang in en tr mix; do
    if [ ! -f "$WORK/imatrix-$lang.dat" ]; then
      echo "== imatrix ($lang)"
      "$IMAT" -m "$MODEL" -f "calib/$lang.txt" -o "$WORK/imatrix-$lang.dat" -c "$CTX" --chunks 100 -ngl "$NGL" \
        >"$WORK/imatrix-$lang.log" 2>&1 || { tail -20 "$WORK/imatrix-$lang.log" >&2; exit 1; }
    fi
  done
  build_eval "q3_k_s+imatrix_none" "$out"
  for lang in en tr mix; do
    build_eval "q3_k_s+imatrix_$lang" "$out" --imatrix "$WORK/imatrix-$lang.dat"
  done
}

stage_report() { python3 pilot_analyze.py report; }

STAGES=("$@")
[ ${#STAGES[@]} -gt 0 ] || STAGES=(check base profile alloc calib report)
for s in "${STAGES[@]}"; do
  [ "$s" = check ] || [ -f "$WORK/n_layers" ] || stage_check
  "stage_$s"
done
