#!/usr/bin/env python3
"""imatrix dosyalarını okur ve seçili FFN nöronlarının önemini artırır (Y1).

llama-imatrix varsayılan olarak GGUF biçiminde yazar (uzantı .dat olsa bile):
her tensör için "<ad>.in_sum2" (girdi kanalı başına aktivasyon kareleri toplamı) ve
"<ad>.counts". ffn_down tensörünün girdi kanalları FFN nöronlarıdır; dolayısıyla
nöron j'nin önemi blk.L.ffn_down.weight.in_sum2[j] değeridir.

Alt komutlar:
  boost SRC DST SET GAMMA [SEED]   SET: tr | en | fi | ... | rand (rand: TR kadar rastgele nöron)
                                   seçili nöronların in_sum2 değerini GAMMA ile çarpar
  compare A B                      iki imatrix'in ffn_down önemlerini dile özgü nöronlarda karşılaştırır
"""
import shutil
import struct
import sys

import numpy as np

from common import load_neurons, random_like


def is_gguf(path):
    with open(path, "rb") as f:
        return f.read(4) == b"GGUF"


def read_legacy(path):
    out = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<i", f.read(4))
        for _ in range(n):
            (ln,) = struct.unpack("<i", f.read(4))
            name = f.read(ln).decode()
            ncall, nval = struct.unpack("<ii", f.read(8))
            vals = np.frombuffer(f.read(4 * nval), dtype=np.float32).copy()
            out[name] = (ncall, vals)
        rest = f.read()
    return out, rest


def write_legacy(path, entries, rest):
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(entries)))
        for name, (ncall, vals) in entries.items():
            b = name.encode()
            f.write(struct.pack("<i", len(b)) + b + struct.pack("<ii", ncall, len(vals)))
            f.write(vals.astype(np.float32).tobytes())
        f.write(rest)


def importance(path):
    """{tensör adı: kanal başına önem} (ortalama aktivasyon karesi)."""
    if is_gguf(path):
        import gguf

        r = gguf.GGUFReader(path)
        t = {x.name: x for x in r.tensors}
        out = {}
        for name in t:
            if name.endswith(".in_sum2"):
                base = name[: -len(".in_sum2")]
                cnt = np.asarray(t[base + ".counts"].data, dtype=np.float64).reshape(-1)
                s2 = np.asarray(t[name].data, dtype=np.float64).reshape(len(cnt), -1)
                out[base] = (s2 / np.clip(cnt[:, None], 1, None)).mean(axis=0)
        return out
    entries, _ = read_legacy(path)
    return {k: v[1].astype(np.float64) / max(v[0], 1) for k, v in entries.items()}


def neuron_set(which, seed=0):
    nd = load_neurons()
    langs = list(nd["langs"])
    masks = nd["masks"]
    if which == "rand":
        return random_like(masks[langs.index("tr")], masks.any(axis=0), seed)
    return masks[langs.index(which)]


def boost(src, dst, which, gamma, seed=0):
    sel = neuron_set(which, seed)
    shutil.copyfile(src, dst)
    if is_gguf(dst):
        import gguf

        r = gguf.GGUFReader(dst, "r+")
        t = {x.name: x for x in r.tensors}
        for layer in range(sel.shape[0]):
            arr = t[f"blk.{layer}.ffn_down.weight.in_sum2"].data
            arr.reshape(-1, sel.shape[1])[:, sel[layer]] *= gamma
        r.data.flush()
        del r
    else:
        entries, rest = read_legacy(dst)
        for layer in range(sel.shape[0]):
            ncall, vals = entries[f"blk.{layer}.ffn_down.weight"]
            vals[sel[layer]] *= gamma
        write_legacy(dst, entries, rest)
    print(f"{dst}: {int(sel.sum())} nöron x{gamma}")


def compare(a, b):
    """A/B önem oranı: dile özgü nöronlarda ve diğerlerinde (medyan)."""
    ia, ib = importance(a), importance(b)
    nd = load_neurons()
    langs = list(nd["langs"])
    masks = nd["masks"]
    print("küme,n,medyan_A/B")
    any_spec = masks.any(axis=0)
    sets = {lang: masks[d] for d, lang in enumerate(langs)}
    sets["diğer"] = ~any_spec
    for name, m in sets.items():
        ratios = []
        for layer in range(m.shape[0]):
            key = f"blk.{layer}.ffn_down.weight"
            r = ia[key] / np.clip(ib[key], 1e-12, None)
            ratios.append(r[m[layer]])
        ratios = np.concatenate(ratios)
        print(f"{name},{len(ratios)},{np.median(ratios) if len(ratios) else float('nan'):.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "boost":
        boost(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), int(sys.argv[6]) if len(sys.argv) > 6 else 0)
    elif cmd == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
