"""Analiz betiklerinin ortak yardımcıları: metin parçalama, GGUF ağırlık okuma,
nöron kümeleri ve cihaz seçimi."""
import os
from pathlib import Path

import numpy as np

RES = Path(os.environ.get("RES", "results"))
CALIB = Path("calib")

# GGUF tensör adı (blk.{i}.X.weight) -> Hugging Face modül yolu (model.layers.{i}.Y)
HF_NAMES = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}
GROUPS = {"attn": ["attn_q", "attn_k", "attn_v", "attn_output"], "ffn": ["ffn_gate", "ffn_up", "ffn_down"]}


def device():
    import torch

    if os.environ.get("DEVICE"):
        return os.environ["DEVICE"]
    if torch.cuda.is_available():  # ROCm de "cuda" olarak görünür
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_hf(path, dev, gguf_path=None):
    """Hugging Face modelini yükler. gguf_path verilirse bütün ağırlıklar o F16 GGUF
    dosyasından alınır: yayımlanan GGUF ile Hugging Face ağırlıkları farklı sürümlerden
    olabiliyor (Qwen2.5'te görüldü), analizlerin llama.cpp ölçümleriyle aynı modele
    bakması için bu gerekli. Yalnız dönüşümde ağırlıkları değiştirilmeyen mimariler
    (qwen2) desteklenir; llama mimarisinde Q/K ağırlıkları dönüşümde yeniden dizilir."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from transformers import AutoConfig

    tok = AutoTokenizer.from_pretrained(path)
    if gguf_path:
        # Yalnız yapılandırma gerekir; ağırlıkların hepsi GGUF'tan doldurulur.
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(path)).float()
        loaded = load_gguf_weights(model, gguf_path)
        skip = {"lm_head.weight"} if model.config.tie_word_embeddings else set()
        unset = [n for n, _ in model.named_parameters() if n not in loaded and n not in skip]
        if unset:
            raise SystemExit(f"GGUF'tan doldurulmayan parametreler: {unset[:5]}")
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    return tok, model.to(dev).eval()


def load_gguf_weights(model, gguf_path):
    import torch

    g = Gguf(gguf_path)
    arch = bytes(g.reader.fields["general.architecture"].parts[-1]).decode()
    if arch != "qwen2":
        raise SystemExit(f"GGUF ağırlık yükleme yalnız qwen2 için doğrulandı, bu model: {arch}")
    names = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
             "output.weight": "lm_head.weight"}
    for i in range(g.n_layers()):
        names[f"blk.{i}.attn_norm.weight"] = f"model.layers.{i}.input_layernorm.weight"
        names[f"blk.{i}.ffn_norm.weight"] = f"model.layers.{i}.post_attention_layernorm.weight"
        for t, hf in HF_NAMES.items():
            for kind in ("weight", "bias"):
                names[f"blk.{i}.{t}.{kind}"] = f"model.layers.{i}.{hf}.{kind}"
    state = model.state_dict()
    loaded = set()
    for gname, hname in names.items():
        if gname not in g.tensors or hname not in state:
            continue
        w = torch.from_numpy(g.get(gname))
        if tuple(w.shape) != tuple(state[hname].shape):
            raise SystemExit(f"Boyut uyuşmuyor: {gname} {tuple(w.shape)} / {hname} {tuple(state[hname].shape)}")
        state[hname].copy_(w)
        loaded.add(hname)
    missing = [n for n in g.tensors if n not in names]
    if missing:
        raise SystemExit(f"Eşlenmeyen GGUF tensörleri: {missing[:5]}")
    print(f"{len(loaded)} tensör GGUF'tan yüklendi ({gguf_path})")
    return loaded


def chunks(tok, lang, n_chunks, ctx=512, kind="eval"):
    """calib/eval_{lang}.txt (veya calib/{lang}.txt) metnini ctx tokenlik parçalara böler."""
    path = CALIB / (f"eval_{lang}.txt" if kind == "eval" else f"{lang}.txt")
    ids = tok(path.read_text(), add_special_tokens=False)["input_ids"]
    n = min(n_chunks, len(ids) // ctx)
    if n < n_chunks:
        print(f"uyarı: {path} yalnız {n} parça veriyor (istenen {n_chunks})")
    return [ids[i * ctx:(i + 1) * ctx] for i in range(n)]


class Gguf:
    """GGUF dosyasından tensörleri float32 olarak okur (kuantize türler açılır)."""

    def __init__(self, path):
        import re

        import gguf

        path = str(path)
        parts = [path]
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path)
        if m:  # bölünmüş GGUF: tüm parçalar
            parts = [path[: m.start()] + f"-{i:05d}-of-{m.group(2)}.gguf" for i in range(1, int(m.group(2)) + 1)]
        readers = [gguf.GGUFReader(p) for p in parts]
        self.reader = readers[0]
        self.tensors = {t.name: t for r in readers for t in r.tensors}

    def get(self, name):
        from gguf import GGMLQuantizationType as T
        from gguf.quants import dequantize

        t = self.tensors[name]
        if t.tensor_type in (T.F32, T.F16):
            return np.asarray(t.data, dtype=np.float32).reshape(t.data.shape)
        return dequantize(t.data, t.tensor_type).astype(np.float32)

    def type_name(self, name):
        return self.tensors[name].tensor_type.name

    def n_layers(self):
        return 1 + max(int(n.split(".")[1]) for n in self.tensors if n.startswith("blk."))


def load_neurons(path=None):
    """neurons.py çıktısı: {dil: [katman başına bool maske]} ve istatistikler."""
    data = np.load(path or RES / "neurons.npz", allow_pickle=True)
    return {k: data[k] for k in data.files}


def random_like(masks, exclude, seed):
    """Her katmanda maskedeki kadar nöronu, exclude dışından rastgele seçer."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(masks)
    for layer in range(masks.shape[0]):
        k = int(masks[layer].sum())
        pool = np.flatnonzero(~exclude[layer])
        if k:
            out[layer, rng.choice(pool, size=min(k, len(pool)), replace=False)] = True
    return out
