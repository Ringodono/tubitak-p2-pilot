#!/usr/bin/env bash
# P2 pilotu: Türkçe ve İngilizce (ve Fince kontrol) kuantizasyon hassasiyeti, mekanizma
# analizleri ve dile duyarlı sıkıştırma ön denemeleri.
#
# Kullanım: ./pilot.sh [aşama...]     (varsayılan: v1 aşamaları)
#   llama.cpp aşamaları:
#     check    dosya, araç ve katman sayısı kontrolü
#     base     F16 taban logitleri (her dil)
#     profile  her katmanın dikkat / ileri besleme grubu tek tek LOW türüne; KLD (H1)
#     calib    imatrix yok / en / tr / mix ile Q3_K_S (kalibrasyon dili)
#     share    TR payı %0–100 imatrix taraması, Q3_K_S ve Q2_K (Y3)
#     alloc    dile göre grup seçimi, eşit boyut (H2)
#     keep     analiz için kalıcı kuantize modeller (Q3_K_S none/en/tr, saf Q2_K)
#     boost    Türkçe nöron ağırlıklı imatrix ve kontrolleri, Q3_K_S ve Q2_K (Y1, M2)
#   Python (PyTorch) aşamaları, HF gerekir:
#     neurons  dile özgü FFN nöronları (LAPE) ve aktivasyon istatistikleri (M1)
#     qerror   kuantizasyon hatasının nöronlara dağılımı (M1, M2)
#     repair   yalnız TR nöronlarını onarma, rastgele kontrolle (M1 nedensel) ve token türü (M3)
#     fisher   dile özgü Fisher kestirimi, profille karşılaştırma için (Y2)
#   report   v1 raporu ($RES/pilot_results.md)
# Ortam değişkenleri:
#   MODEL   F16 GGUF (bölünmüş dosyada ilk parça)  (varsayılan: models/qwen2.5-1.5b-instruct-fp16.gguf)
#   HF      Hugging Face yapılandırma/tokenizer dizini (Python aşamaları için)
#   TAG     verilirse çıktı work/$TAG ve results/$TAG altına yazılır
#   LANGS   KLD ölçülen diller (varsayılan: "tr en fi")
#   LAPE_LANGS  nöron analizi dilleri (varsayılan: tr,en,fi,de,ru)
#   BIN, NGL, CHUNKS (20), LOW (q2_K), K (4), PY (python3), LAPE_CHUNKS (64), TORCH_CHUNKS (16)
# Her aşama kaldığı yerden devam eder.
set -euo pipefail

MODEL="${MODEL:-models/qwen2.5-1.5b-instruct-fp16.gguf}"
HF="${HF:-}"
BIN="${BIN:-}"
NGL="${NGL:-99}"
CHUNKS="${CHUNKS:-20}"
LOW="${LOW:-q2_K}"
K="${K:-4}"
PY="${PY:-python3}"
LANGS="${LANGS:-tr en fi}"
LAPE_LANGS="${LAPE_LANGS:-tr,en,fi,de,ru}"
LAPE_CHUNKS="${LAPE_CHUNKS:-64}"
TORCH_CHUNKS="${TORCH_CHUNKS:-16}"
CTX=512
if [ -n "${TAG:-}" ]; then WORK="work/$TAG"; RES="results/$TAG"; else WORK=work; RES=results; fi
export RES
mkdir -p "$WORK/keep" "$RES"

bin() { if [ -n "$BIN" ]; then echo "$BIN/$1"; else echo "$1"; fi; }
QUANT="$(bin llama-quantize)"
PPL="$(bin llama-perplexity)"
IMAT="$(bin llama-imatrix)"
HDR="model,size_mb,lang,kld,kld_err,top1"

size_mb() { echo $(( $(stat -c %s "$1" 2>/dev/null || stat -f %z "$1") / 1048576 )); }
die() { echo "$*" >&2; exit 1; }
need_hf() { [ -n "$HF" ] && [ -f "$HF/config.json" ] || die "HF dizini gerekli (config.json ve tokenizer): HF=$HF"; }

# kld MODEL LANG LOG -> "kld hata top1"
kld() {
  "$PPL" -m "$1" -c "$CTX" --chunks "$CHUNKS" -ngl "$NGL" \
    --kl-divergence-base "$WORK/base_$2.kld" --kl-divergence >"$3" 2>&1 \
    || { tail -20 "$3" >&2; die "llama-perplexity hata verdi, günlük: $3"; }
  "$PY" pilot_analyze.py parse "$3"
}

# missing_langs KEY CSV: CSV'de "KEY,<dil>," satırı olmayan diller
missing_langs() {
  local out=() lang
  for lang in $LANGS; do grep -q "^$1,$lang," "$2" 2>/dev/null || grep -q "^$1,[0-9]*,$lang," "$2" 2>/dev/null || out+=("$lang"); done
  echo "${out[@]:-}"
}

imatrix() { # imatrix METİN_ADI -> $WORK/imatrix-METİN_ADI.dat
  local f="$WORK/imatrix-$1.dat"
  if [ ! -f "$f" ]; then
    echo "== imatrix ($1)" >&2
    "$IMAT" -m "$MODEL" -f "calib/$1.txt" -o "$f" -c "$CTX" --chunks 100 -ngl "$NGL" \
      >"$WORK/imatrix-$1.log" 2>&1 || { tail -20 "$WORK/imatrix-$1.log" >&2; die "llama-imatrix hata verdi"; }
  fi
  echo "$f"
}

# build_eval AD CSV FTYPE [llama-quantize seçenekleri...]: eksik diller için KLD ölçer
build_eval() {
  local name="$1" out="$2" ftype="$3"; shift 3
  [ -f "$out" ] || echo "$HDR" >"$out"
  local miss; miss="$(missing_langs "$name" "$out")"
  [ -n "$miss" ] || return 0
  echo "== $name"
  "$QUANT" "$@" "$MODEL" "$WORK/tmp_eval.gguf" "$ftype" >"$WORK/q.log" 2>&1 || { tail -20 "$WORK/q.log" >&2; die "llama-quantize hata verdi"; }
  local mb lang v e t; mb="$(size_mb "$WORK/tmp_eval.gguf")"
  for lang in $miss; do
    read -r v e t < <(kld "$WORK/tmp_eval.gguf" "$lang" "$WORK/kld.log")
    echo "$name,$mb,$lang,$v,$e,$t" >>"$out"
  done
  rm -f "$WORK/tmp_eval.gguf"
}

stage_check() {
  [ -f "$MODEL" ] || die "Eksik model: $MODEL (README)"
  for lang in $LANGS; do [ -f "calib/eval_$lang.txt" ] || die "Eksik: calib/eval_$lang.txt (python3 make_calib.py)"; done
  for f in calib/tr.txt calib/en.txt calib/mix.txt; do [ -f "$f" ] || die "Eksik: $f (python3 make_calib.py)"; done
  for c in "$QUANT" "$PPL" "$IMAT"; do command -v "$c" >/dev/null || die "Bulunamadı: $c (BIN değişkenini ayarla)"; done
  "$PY" -c "
import sys
sys.path.insert(0, 'analysis')
from common import Gguf
print(Gguf(sys.argv[1]).n_layers())" "$MODEL" >"$WORK/n_layers"
  echo "Model: $MODEL, katman: $(cat "$WORK/n_layers"), çıktı: $RES, diller: $LANGS"
}

stage_base() {
  local lang
  for lang in $LANGS; do
    if [ ! -f "$WORK/base_$lang.kld" ]; then
      echo "== F16 taban logitleri ($lang)"
      "$PPL" -m "$MODEL" -f "calib/eval_$lang.txt" -c "$CTX" --chunks "$CHUNKS" -ngl "$NGL" \
        --kl-divergence-base "$WORK/base_$lang.kld" >"$WORK/base_$lang.log" 2>&1 \
        || { tail -20 "$WORK/base_$lang.log" >&2; die "taban logitleri üretilemedi"; }
    fi
  done
}

stage_profile() {
  local n_layers out="$RES/profile.csv" i g lang miss v e t
  n_layers="$(cat "$WORK/n_layers")"
  [ -f "$out" ] || echo "layer,group,lang,kld,kld_err,top1" >"$out"
  for ((i = -1; i < n_layers; i++)); do
    for g in attn ffn; do
      if [ "$i" = -1 ]; then [ "$g" = attn ] || continue; key="-1,none"; else key="$i,$g"; fi
      miss=""
      for lang in $LANGS; do grep -q "^$key,$lang," "$out" || miss="$miss $lang"; done
      [ -n "$miss" ] || continue
      if [ "$i" = -1 ]; then
        echo "== Q8_0 tabanı"
        "$QUANT" "$MODEL" "$WORK/tmp.gguf" Q8_0 >"$WORK/q.log" 2>&1 || { tail -20 "$WORK/q.log" >&2; die "llama-quantize hata verdi"; }
      else
        echo "== katman $i / $g -> $LOW"
        "$QUANT" --tensor-type "blk\.$i\.${g}_.*=$LOW" "$MODEL" "$WORK/tmp.gguf" Q8_0 >"$WORK/q.log" 2>&1 \
          || { tail -20 "$WORK/q.log" >&2; die "llama-quantize hata verdi"; }
      fi
      for lang in $miss; do
        read -r v e t < <(kld "$WORK/tmp.gguf" "$lang" "$WORK/kld.log")
        echo "$key,$lang,$v,$e,$t" >>"$out"
      done
    done
  done
  rm -f "$WORK/tmp.gguf"
}

stage_alloc() {
  local out="$RES/alloc.csv"
  build_eval q3_k_s "$out" Q3_K_S
  # sel: seçim ölçütü, im: imatrix dili (boş = imatrix yok)
  local arm sel im name pat
  for arm in en: tr: tr_excess: en:en tr:tr tr_excess:tr; do
    sel="${arm%%:*}" im="${arm#*:}"
    local args=()
    while read -r pat; do args+=(--tensor-type "$pat=q5_K"); done < <("$PY" pilot_analyze.py select "$sel" "$K")
    name="q3_k_s+${sel}_aware"
    if [ -n "$im" ]; then
      args+=(--imatrix "$(imatrix "$im")")
      name="$name+imatrix_$im"
    fi
    build_eval "$name" "$out" Q3_K_S "${args[@]}"
  done
}

stage_calib() {
  local out="$RES/calib.csv" lang
  build_eval "q3_k_s+imatrix_none" "$out" Q3_K_S
  for lang in en tr mix; do
    build_eval "q3_k_s+imatrix_$lang" "$out" Q3_K_S --imatrix "$(imatrix "$lang")"
  done
}

stage_share() {
  local out="$RES/share.csv" s ft im
  for s in 000 025 050 075 100; do
    [ -f "calib/tr$s.txt" ] || die "Eksik: calib/tr$s.txt (python3 make_calib.py)"
    im="$(imatrix "tr$s")"
    for ft in Q3_K_S Q2_K; do
      build_eval "$(echo "$ft" | tr 'A-Z' 'a-z')+tr$s" "$out" "$ft" --imatrix "$im"
    done
  done
}

stage_keep() {
  local k="$WORK/keep"
  [ -f "$k/q3_k_s_none.gguf" ] || "$QUANT" "$MODEL" "$k/q3_k_s_none.gguf" Q3_K_S >"$WORK/q.log" 2>&1
  for lang in en tr; do
    [ -f "$k/q3_k_s_$lang.gguf" ] || "$QUANT" --imatrix "$(imatrix "$lang")" "$MODEL" "$k/q3_k_s_$lang.gguf" Q3_K_S >"$WORK/q.log" 2>&1
  done
  [ -f "$k/q2_k_pure.gguf" ] || "$QUANT" --pure "$MODEL" "$k/q2_k_pure.gguf" Q2_K >"$WORK/q.log" 2>&1
  ls -la "$k"
}

stage_neurons() {
  need_hf
  [ -f "$RES/neurons.npz" ] && return 0
  "$PY" analysis/neurons.py --hf "$HF" --gguf "$MODEL" --langs "$LAPE_LANGS" --chunks "$LAPE_CHUNKS"
}

stage_qerror() {
  [ -f "$RES/neurons.npz" ] || die "Önce neurons aşaması"
  stage_keep >/dev/null
  [ -f "$RES/qerror.csv" ] && return 0
  local k="$WORK/keep"
  "$PY" analysis/qerror.py --f16 "$MODEL" imatrix_none="$k/q3_k_s_none.gguf" imatrix_en="$k/q3_k_s_en.gguf" imatrix_tr="$k/q3_k_s_tr.gguf"
  "$PY" analysis/imatrix_tools.py compare "$(imatrix tr)" "$(imatrix en)" | tee "$RES/imatrix_compare.csv"
}

stage_repair() {
  need_hf
  [ -f "$RES/neurons.npz" ] || die "Önce neurons aşaması"
  stage_keep >/dev/null
  [ -f "$RES/repair.csv" ] && return 0
  "$PY" analysis/repair.py --hf "$HF" --gguf "$MODEL" --quant "$WORK/keep/q3_k_s_en.gguf" \
    --langs "$(echo $LANGS | tr ' ' ',')" --chunks "$TORCH_CHUNKS" --seeds 3
}

stage_fisher() {
  need_hf
  stage_keep >/dev/null
  [ -f "$RES/fisher.csv" ] && return 0
  "$PY" analysis/fisher.py --hf "$HF" --gguf "$MODEL" --q2 "$WORK/keep/q2_k_pure.gguf" \
    --langs "$(echo $LANGS | tr ' ' ',')" --chunks "$TORCH_CHUNKS"
}

stage_boost() {
  [ -f "$RES/neurons.npz" ] || die "Önce neurons aşaması"
  local out="$RES/boost.csv" ft lft arm base set gamma seed im src
  for ft in Q3_K_S Q2_K; do
    lft="$(echo "$ft" | tr 'A-Z' 'a-z')"
    for base in en tr tr050; do
      build_eval "$lft+imatrix_$base" "$out" "$ft" --imatrix "$(imatrix "$base")"
    done
    # taban:küme:gamma:tohum  (rand = TR kadar rastgele nöron; en/fi = başka dilin nöronları, kontrol)
    for arm in en:tr:4:0 en:tr:16:0 en:tr:64:0 en:rand:16:0 en:rand:16:1 en:en:16:0 en:fi:16:0 tr050:tr:16:0 tr:tr:16:0; do
      IFS=: read -r base set gamma seed <<<"$arm"
      src="$(imatrix "$base")"
      im="$WORK/imatrix-$base+$set-x$gamma-s$seed.dat"
      [ -f "$im" ] || "$PY" analysis/imatrix_tools.py boost "$src" "$im" "$set" "$gamma" "$seed"
      build_eval "$lft+imatrix_$base+boost_${set}_x${gamma}_s$seed" "$out" "$ft" --imatrix "$im"
    done
  done
}

stage_report() { "$PY" pilot_analyze.py report; }

STAGES=("$@")
[ ${#STAGES[@]} -gt 0 ] || STAGES=(check base profile alloc calib report)
for s in "${STAGES[@]}"; do
  [ "$s" = check ] || [ -f "$WORK/n_layers" ] || stage_check
  echo "######## $s ($(date +%H:%M))"
  "stage_$s"
done
