#!/usr/bin/env python3
"""M1 için dile özgü FFN nöronları (LAPE, Tang vd. ACL 2024) ve dil başına
aktivasyon istatistikleri.

Her dil için calib/{dil}.txt üzerinde:
  p[d, katman, nöron]  = P(gate_proj çıktısı > 0)  (SiLU > 0 ile aynı)
  a2[d, katman, nöron] = E[h^2], h = down_proj girdisi (act(gate) * up)
LAPE: en etkin olasılığı %95'lik dilimin altında kalan nöronlar elenir; kalanlardan
dil dağılımı entropisi en düşük %1 seçilir; nöron, olasılığı eşiği aşan dillere atanır.

Kullanım: python3 analysis/neurons.py --hf DİZİN [--langs tr,en,fi,de,ru] [--chunks 64]
Çıktı: $RES/neurons.npz, $RES/neurons_layers.csv
"""
import argparse
import csv
import time

import numpy as np
import torch

from common import RES, chunks, device, load_hf


def collect(model, tok, lang, n_chunks, dev):
    layers = model.model.layers
    n_inter = layers[0].mlp.gate_proj.out_features
    acc = torch.float32 if dev == "mps" else torch.float64  # MPS float64 desteklemiyor
    pos = torch.zeros(len(layers), n_inter, dtype=acc, device=dev)
    sq = torch.zeros_like(pos)
    count = [0]
    hooks = []

    # Kancalar None döndürmeli: değer döndüren kanca katmanın girdisini/çıktısını değiştirir.
    def on_gate(i):
        def hook(module, inp, out):
            pos[i].add_((out > 0).sum(dim=(0, 1)).to(acc))
        return hook

    def on_down(i):
        def hook(module, inp):
            sq[i].add_((inp[0].to(acc) ** 2).sum(dim=(0, 1)))
        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.mlp.gate_proj.register_forward_hook(on_gate(i)))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(on_down(i)))
    with torch.no_grad():
        for ids in chunks(tok, lang, n_chunks, kind="calib"):
            model(torch.tensor([ids], device=dev))
            count[0] += len(ids)
    for h in hooks:
        h.remove()
    return (pos / count[0]).float().cpu().numpy(), (sq / count[0]).float().cpu().numpy(), count[0]


def lape(p, top_rate=0.01, filter_rate=0.95, bar_rate=0.95):
    """p: (diller, katman, nöron). Dile özgü maskeler (diller, katman, nöron) döner."""
    flat = p.reshape(p.shape[0], -1)
    tau_filter = np.quantile(flat, filter_rate)
    norm = flat / np.clip(flat.sum(axis=0, keepdims=True), 1e-12, None)
    ent = -(norm * np.log(np.clip(norm, 1e-12, None))).sum(axis=0)
    ent[flat.max(axis=0) < tau_filter] = np.inf
    k = int(round(flat.shape[1] * top_rate))
    selected = np.argsort(ent)[:k]
    bar = np.quantile(flat, bar_rate)
    masks = np.zeros_like(flat, dtype=bool)
    for d in range(flat.shape[0]):
        masks[d, selected] = flat[d, selected] > bar
    return masks.reshape(p.shape), ent.reshape(p.shape[1:]), float(tau_filter), float(bar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="yapılandırma ve tokenizer için Hugging Face dizini")
    ap.add_argument("--gguf", required=True, help="ağırlıkların alınacağı F16 GGUF (llama.cpp ölçümleriyle aynı model)")
    ap.add_argument("--langs", default="tr,en,fi,de,ru")
    ap.add_argument("--chunks", type=int, default=64)
    args = ap.parse_args()
    langs = args.langs.split(",")
    dev = device()
    tok, model = load_hf(args.hf, dev, args.gguf)
    print(f"cihaz: {dev}, katman: {len(model.model.layers)}")

    ps, a2s = [], []
    for lang in langs:
        t0 = time.time()
        p, a2, n = collect(model, tok, lang, args.chunks, dev)
        ps.append(p)
        a2s.append(a2)
        print(f"{lang}: {n} token, {time.time() - t0:.0f} sn")
    p, a2 = np.stack(ps), np.stack(a2s)
    masks, ent, tau, bar = lape(p)

    RES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RES / "neurons.npz", langs=np.array(langs), p=p, a2=a2, entropy=ent,
                        masks=masks, tau=tau, bar=bar)
    with open(RES / "neurons_layers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "n_specific"] + [f"n_{lang}" for lang in langs])
        for layer in range(p.shape[1]):
            w.writerow([layer, int(masks[:, layer].any(axis=0).sum())] + [int(masks[d, layer].sum()) for d in range(len(langs))])
    total = {lang: int(masks[d].sum()) for d, lang in enumerate(langs)}
    print(f"dile özgü nöronlar: {total} (eşik {bar:.3f})")


if __name__ == "__main__":
    main()
