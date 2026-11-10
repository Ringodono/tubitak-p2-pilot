# P2 pilotu: Türkçe ve İngilizce kuantizasyon hassasiyeti

Bu pilot, Qwen2.5-1.5B-Instruct'ta her katmanın dikkat (attn) ve ileri besleme (ffn) gruplarını tek tek Q2_K'ya düşürür. Geri kalan tensörler Q8_0'da kalır. Her model için F16 modele göre KL sapması Türkçe ve İngilizce metinde ölçülür. Ardından üç şey yapılır:

- Türkçe ve İngilizce hassasiyet sıralaması karşılaştırılır (H1).
- Dile göre bit dağıtımı aynı dosya boyutunda denenir (H2 ön denemesi).
- imatrix kalibrasyonu farklı dillerle denenir.

Süre RX 7700 XT'de yaklaşık 1–2 saat. Diskte yaklaşık 10 GB boş yer gerekir; taban logit dosyalarının her biri yaklaşık 1,6 GB.

## 1. Kurulum (Linux)

En kolay yol llama.cpp'nin hazır Vulkan ikililerini kullanmak; derleme gerekmez:

```bash
sudo apt update
sudo apt install python3-pip libvulkan1 mesa-vulkan-drivers vulkan-tools unzip curl
vulkaninfo --summary          # RX 7700 XT listede görünmeli

mkdir -p ~/llama.cpp && cd ~/llama.cpp
curl -L -o llama.tar.gz https://github.com/ggml-org/llama.cpp/releases/download/b12244/llama-b12244-bin-ubuntu-vulkan-x64.tar.gz
tar xzf llama.tar.gz
export BIN=$(dirname $(find ~/llama.cpp -name llama-quantize -type f | head -1))
echo $BIN                     # bir klasör yolu yazmalı

pip install datasets gguf huggingface_hub
```

Kaynaktan derlemek istersen Ubuntu 24.04 veya üstü gerekir, çünkü eski sürümlerde `glslc` paketi yok:

```bash
sudo apt install build-essential cmake git libvulkan-dev glslc spirv-headers
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp-src && cd ~/llama.cpp-src
cmake -B build -DGGML_VULKAN=ON && cmake --build build --config Release -j
export BIN=~/llama.cpp-src/build/bin
```

Ekran kartı çalışmazsa pilot yalnız işlemciyle de çalışır: `NGL=0 ./pilot.sh`. Bu yol daha yavaştır.

## 2. Model

```bash
cd ~/tubitak-p2-pilot          # bu klasör
hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-fp16.gguf --local-dir models
# eski sürümlerde: huggingface-cli download ... (aynı argümanlar)
```

## 3. Metinler

```bash
python3 make_calib.py
```

Bu komut `calib/` altına beş dosya yazar: kalibrasyon için `en.txt`, `tr.txt`, `mix.txt`; değerlendirme için `eval_en.txt`, `eval_tr.txt`. Değerlendirme metinleri kalibrasyonla aynı Wikipedia akışından ama sonra alındığı için iki küme ayrıktır.

## 4. Çalıştırma

```bash
./pilot.sh
```

Aşamalar sırayla çalışır. Yarıda kesilirse aynı komut kaldığı yerden devam eder.

| Aşama | Ne yapıyor | Çıktı |
| --- | --- | --- |
| `check` | Dosyaları, araçları ve katman sayısını kontrol eder | `work/n_layers` |
| `base` | F16 taban logitleri (TR, EN) | `work/base_*.kld` |
| `profile` | Q8_0 tabanı + 28 katman × {attn, ffn} = 56 model | `results/profile.csv` |
| `alloc` | Q3_K_S tabanı; EN'e ve TR'ye göre en hassas 4 attn + 4 ffn grubu Q5_K'ya yükseltilir (eşit boyut) | `results/alloc.csv` |
| `calib` | Q3_K_S; imatrix yok / EN / TR / karışık | `results/calib.csv` |
| `report` | Analiz ve rapor | `results/pilot_results.md` |

Tek bir aşama da çalıştırılabilir, örneğin `./pilot.sh report`. Ayarlar ortam değişkenleriyle değiştirilir: `MODEL`, `BIN`, `NGL`, `CHUNKS`, `LOW`, `K`. Hepsinin açıklaması `pilot.sh`'nin başında.

## 5. Sonuç

`results/pilot_results.md` şu bilgileri içerir:

- **H1:** Türkçe–İngilizce Spearman ρ, en hassas %20'lik grupların örtüşmesi ve eşiğin sağlanıp sağlanmadığı.
- **Sıra farkları:** Türkçede belirgin biçimde daha hassas (ya da daha az hassas) çıkan katmanlar.
- **H2 ön denemesi:** Aynı boyutta TR'ye göre ve EN'e göre dağıtımın Türkçe ve İngilizce KLD değerleri. TR'ye göre dağıtım Türkçe KLD'yi daha çok düşürüyorsa bu, H2 için ön kanıttır.
- **Kalibrasyon dili:** imatrix diline göre Türkçe ve İngilizce KLD.

Bu tabloyu proje raporundaki "Pilot sonuçları" kısmına ekleyebilirsin; istersen sonuçları bana yapıştır, ben yorumlayıp rapora yazarım.

## Notlar

- KLD küçükse model F16'ya o kadar yakındır. Hassasiyet, o grup düşürüldüğünde KLD'nin Q8_0 tabanına göre ne kadar arttığı olarak ölçülür.
- Bu pilotta 20 parça × 512 token kullanılıyor. Sonuçlar gürültülü görünürse `CHUNKS=40` ile yeniden çalıştır. Önce `work/base_*.kld` dosyalarını silmen gerekir.
- Tür adları ggml adlarıdır (`q2_K`, `q5_K`). Tensör desenleri regex ile kısmi eşleşme olarak aranır (`blk\.5\.attn_.*`).
