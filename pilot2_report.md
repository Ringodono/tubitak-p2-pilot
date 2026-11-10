# P2 pilot 2 raporu

Modeller: v1 (results/), eurollm-1.7b, qwen2.5-1.5b, qwen2.5-3b

## 1. Olgu: Türkçe kırılganlığı ve katman profili

| Model | TR/EN KLD (Q3_K_S, imatrix yok) | FI/EN | Profil ρ (TR–EN) | TR/EN ilk %20 katman | orta | son %20 |
| --- | --- | --- | --- | --- | --- | --- |
| v1 (results/) | 2.72 | - | 0.874 | 3.67 | 1.90 | 2.72 |
| eurollm-1.7b | 1.30 | 1.46 | 0.939 | 1.28 | 1.09 | 1.07 |
| qwen2.5-1.5b | 2.72 | 3.10 | 0.874 | 3.67 | 1.90 | 2.72 |
| qwen2.5-3b | 2.54 | 3.33 | 0.897 | 4.00 | 2.00 | 2.31 |

## 2. Mekanizma (M1–M3)

### qwen2.5-1.5b: dile özgü nöronlar (LAPE)

Toplam: tr 954, en 69, fi 1324, de 384, ru 235

Katman başına TR nöron sayısı ile TR/EN KLD oranı arasında Spearman ρ = **0.216** (pozitifse U şekli Türkçe nöronların yerleşimiyle örtüşüyor).

### qwen2.5-1.5b: kuantizasyon hatasının nöronlara dağılımı

Enerji payı: Türkçe çıktı hata enerjisinin bu kümeden gelen kısmı (katman ortalaması). Yoğunlaşma = enerji payı / nöron payı (1 = eşit dağılım).

| Model | Küme | Nöron | Göreli hata | TR enerji payı | TR yoğunlaşma | EN enerji payı |
| --- | --- | --- | --- | --- | --- | --- |
| imatrix_en | de | 384 | 0.0222 | 0.0075 | 4.9 | 0.0013 |
| imatrix_en | en | 69 | 0.0229 | 0.0004 | 1.5 | 0.0004 |
| imatrix_en | fi | 1324 | 0.0216 | 0.0278 | 5.3 | 0.0050 |
| imatrix_en | rand | 954 | 0.0217 | 0.0030 | 0.8 | 0.0038 |
| imatrix_en | ru | 235 | 0.0233 | 0.0033 | 3.5 | 0.0010 |
| imatrix_en | tr | 954 | 0.0218 | 0.0892 | 23.4 | 0.0036 |
| imatrix_none | de | 384 | 0.0235 | 0.0072 | 4.7 | 0.0011 |
| imatrix_none | en | 69 | 0.0245 | 0.0004 | 1.6 | 0.0005 |
| imatrix_none | fi | 1324 | 0.0229 | 0.0291 | 5.5 | 0.0056 |
| imatrix_none | rand | 954 | 0.0230 | 0.0030 | 0.8 | 0.0038 |
| imatrix_none | ru | 235 | 0.0243 | 0.0033 | 3.5 | 0.0011 |
| imatrix_none | tr | 954 | 0.0230 | 0.0893 | 23.5 | 0.0036 |
| imatrix_tr | de | 384 | 0.0222 | 0.0032 | 2.1 | 0.0011 |
| imatrix_tr | en | 69 | 0.0239 | 0.0004 | 1.4 | 0.0006 |
| imatrix_tr | fi | 1324 | 0.0216 | 0.0125 | 2.4 | 0.0050 |
| imatrix_tr | rand | 954 | 0.0220 | 0.0035 | 0.9 | 0.0039 |
| imatrix_tr | ru | 235 | 0.0233 | 0.0019 | 2.0 | 0.0010 |
| imatrix_tr | tr | 954 | 0.0211 | 0.0309 | 8.1 | 0.0022 |

### qwen2.5-1.5b: imatrix önemi, Türkçe / İngilizce kalibrasyon (medyan oran)

| Nöron kümesi | n | TR imatrix / EN imatrix |
| --- | --- | --- |
| tr | 954 | 10.0560 |
| en | 69 | 0.4142 |
| fi | 1324 | 1.7017 |
| de | 384 | 1.5325 |
| ru | 235 | 1.3272 |
| diğer | 248372 | 0.9538 |

### qwen2.5-1.5b: nedensel test, yalnız FFN kuantize (Q3_K_S, imatrix EN), nöron kümesi onarılıyor

| Yapılandırma | Onarılan nöron | KLD tr (değişim) | KLD en (değişim) | KLD fi (değişim) |
| --- | --- | --- | --- | --- |
| quant | 0 | 0.3027 | 0.0978 | 0.3593 |
| restore_tr | 954 | 0.2724 (-10.0%) | 0.0972 (-0.6%) | 0.3516 (-2.2%) |
| restore_en | 69 | 0.3032 (+0.2%) | 0.0978 (+0.0%) | 0.3596 (+0.1%) |
| restore_fi | 1324 | 0.2932 (-3.2%) | 0.0970 (-0.8%) | 0.3039 (-15.4%) |
| restore_de | 384 | 0.2984 (-1.4%) | 0.0973 (-0.5%) | 0.3585 (-0.2%) |
| restore_ru | 235 | 0.3019 (-0.3%) | 0.0978 (+0.0%) | 0.3583 (-0.3%) |
| restore_rand0 | 954 | 0.2997 (-1.0%) | 0.0971 (-0.8%) | 0.3572 (-0.6%) |
| restore_rand1 | 954 | 0.2998 (-1.0%) | 0.0973 (-0.5%) | 0.3566 (-0.7%) |
| restore_rand2 | 954 | 0.3006 (-0.7%) | 0.0971 (-0.7%) | 0.3562 (-0.9%) |

TR onarımı − rastgele onarım (Türkçe token başına KLD farkı): **-0.02767** (%95 GA -0.03184 … -0.02378); GA sıfırın altındaysa TR nöronları rastgeleden anlamlı fazla onarıyor.

### qwen2.5-1.5b: token türüne göre KLD (M3, yalnız FFN kuantize)

| Dil | Kelime başı | Kelime içi devam | Diğer | Devam / baş |
| --- | --- | --- | --- | --- |
| tr | 0.3626 | 0.3401 | 0.1292 | 0.94 |
| en | 0.0889 | 0.1490 | 0.0498 | 1.68 |
| fi | 0.3190 | 0.4243 | 0.1532 | 1.33 |

### qwen2.5-3b: dile özgü nöronlar (LAPE)

Toplam: tr 1501, en 178, fi 2139, de 576, ru 432

Katman başına TR nöron sayısı ile TR/EN KLD oranı arasında Spearman ρ = **-0.250** (pozitifse U şekli Türkçe nöronların yerleşimiyle örtüşüyor).

### qwen2.5-3b: kuantizasyon hatasının nöronlara dağılımı

Enerji payı: Türkçe çıktı hata enerjisinin bu kümeden gelen kısmı (katman ortalaması). Yoğunlaşma = enerji payı / nöron payı (1 = eşit dağılım).

| Model | Küme | Nöron | Göreli hata | TR enerji payı | TR yoğunlaşma | EN enerji payı |
| --- | --- | --- | --- | --- | --- | --- |
| imatrix_en | de | 576 | 0.0224 | 0.0055 | 3.8 | 0.0012 |
| imatrix_en | en | 178 | 0.0223 | 0.0002 | 0.5 | 0.0007 |
| imatrix_en | fi | 2139 | 0.0219 | 0.0229 | 4.2 | 0.0051 |
| imatrix_en | rand | 1501 | 0.0218 | 0.0027 | 0.7 | 0.0038 |
| imatrix_en | ru | 432 | 0.0232 | 0.0045 | 4.1 | 0.0011 |
| imatrix_en | tr | 1501 | 0.0220 | 0.0868 | 22.9 | 0.0033 |
| imatrix_none | de | 576 | 0.0236 | 0.0049 | 3.4 | 0.0010 |
| imatrix_none | en | 178 | 0.0240 | 0.0003 | 0.6 | 0.0008 |
| imatrix_none | fi | 2139 | 0.0231 | 0.0222 | 4.1 | 0.0050 |
| imatrix_none | rand | 1501 | 0.0231 | 0.0027 | 0.7 | 0.0038 |
| imatrix_none | ru | 432 | 0.0242 | 0.0044 | 4.0 | 0.0011 |
| imatrix_none | tr | 1501 | 0.0231 | 0.0853 | 22.5 | 0.0030 |
| imatrix_tr | de | 576 | 0.0223 | 0.0027 | 1.9 | 0.0010 |
| imatrix_tr | en | 178 | 0.0237 | 0.0003 | 0.7 | 0.0010 |
| imatrix_tr | fi | 2139 | 0.0218 | 0.0117 | 2.2 | 0.0047 |
| imatrix_tr | rand | 1501 | 0.0221 | 0.0033 | 0.9 | 0.0038 |
| imatrix_tr | ru | 432 | 0.0230 | 0.0022 | 2.0 | 0.0010 |
| imatrix_tr | tr | 1501 | 0.0211 | 0.0333 | 8.8 | 0.0019 |

### qwen2.5-3b: imatrix önemi, Türkçe / İngilizce kalibrasyon (medyan oran)

| Nöron kümesi | n | TR imatrix / EN imatrix |
| --- | --- | --- |
| tr | 1501 | 10.5484 |
| en | 178 | 0.2279 |
| fi | 2139 | 1.6876 |
| de | 576 | 1.3347 |
| ru | 432 | 1.5395 |
| diğer | 392325 | 0.9527 |

## 3. Yöntem ön denemeleri (Y1–Y3)

### v1 (results/): Kalibrasyon dili

Q3_K_S.

| Model | MB | KLD tr | KLD en | TR değişim (ref.) |
| --- | --- | --- | --- | --- |
| q3_k_s+imatrix_none | 821 | 0.5244 | 0.1925 | +16.7% |
| q3_k_s+imatrix_en | 821 | 0.4494 | 0.1509 | +0.0% |
| q3_k_s+imatrix_tr | 821 | 0.4039 | 0.1539 | -10.1% |
| q3_k_s+imatrix_mix | 821 | 0.4120 | 0.1515 | -8.3% |

### v1 (results/): Dile göre bit dağıtımı (eşit boyut)

Taban Q3_K_S, seçilen 8 grup Q5_K.

| Model | MB | KLD tr | KLD en | TR değişim (ref.) |
| --- | --- | --- | --- | --- |
| q3_k_s | 821 | 0.5244 | 0.1925 | +67.7% |
| q3_k_s+en_aware | 867 | 0.3478 | 0.1411 | +11.3% |
| q3_k_s+tr_aware | 867 | 0.3336 | 0.1411 | +6.7% |
| q3_k_s+tr_excess_aware | 867 | 0.3568 | 0.1498 | +14.1% |
| q3_k_s+en_aware+imatrix_en | 867 | 0.3126 | 0.1144 | +0.0% |
| q3_k_s+tr_aware+imatrix_tr | 867 | 0.2727 | 0.1218 | -12.8% |
| q3_k_s+tr_excess_aware+imatrix_tr | 867 | 0.2744 | 0.1237 | -12.2% |

### eurollm-1.7b: Kalibrasyonda Türkçe payı (Y3)

trPPP = imatrix metninde %PPP Türkçe, gerisi İngilizce.

| Model | MB | KLD tr | KLD en | KLD fi | TR değişim (ref.) |
| --- | --- | --- | --- | --- | --- |
| q3_k_s+tr000 | 779 | 0.4598 | 0.3172 | 0.4794 | +0.0% |
| q2_k+tr000 | 686 | 0.9227 | 0.5685 | 1.0874 | +0.0% |
| q3_k_s+tr025 | 779 | 0.4543 | 0.3228 | 0.4830 | -1.2% |
| q2_k+tr025 | 686 | 0.8947 | 0.5716 | 1.0703 | -3.0% |
| q3_k_s+tr050 | 779 | 0.4466 | 0.3175 | 0.4758 | -2.9% |
| q2_k+tr050 | 686 | 0.8602 | 0.5618 | 1.0314 | -6.8% |
| q3_k_s+tr075 | 779 | 0.4514 | 0.3218 | 0.4790 | -1.8% |
| q2_k+tr075 | 686 | 0.8813 | 0.5671 | 1.0572 | -4.5% |
| q3_k_s+tr100 | 779 | 0.4411 | 0.3232 | 0.4772 | -4.1% |
| q2_k+tr100 | 686 | 0.8603 | 0.5593 | 1.0239 | -6.8% |

### eurollm-1.7b: Kalibrasyon dili

Q3_K_S.

| Model | MB | KLD tr | KLD en | KLD fi | TR değişim (ref.) |
| --- | --- | --- | --- | --- | --- |
| q3_k_s+imatrix_none | 779 | 0.5647 | 0.4328 | 0.6333 | +22.8% |
| q3_k_s+imatrix_en | 779 | 0.4599 | 0.3201 | 0.4861 | +0.0% |
| q3_k_s+imatrix_tr | 779 | 0.4487 | 0.3271 | 0.4744 | -2.4% |
| q3_k_s+imatrix_mix | 779 | 0.4664 | 0.3213 | 0.4918 | +1.4% |

### qwen2.5-1.5b: Türkçe nöron ağırlıklı imatrix (Y1)

Referans: aynı türde imatrix_en. rand = TR kadar rastgele nöron; en/fi = başka dilin nöronları (kontrol).

| Model | MB | KLD tr | KLD en | KLD fi | TR değişim (ref.) |
| --- | --- | --- | --- | --- | --- |
| q3_k_s+imatrix_en | 821 | 0.4494 | 0.1509 | 0.5302 | +0.0% |
| q3_k_s+imatrix_tr | 821 | 0.4039 | 0.1539 | 0.4956 | -10.1% |
| q3_k_s+imatrix_tr050 | 821 | 0.4094 | 0.1522 | 0.5002 | -8.9% |
| q3_k_s+imatrix_en+boost_tr_x4_s0 | 821 | 0.4396 | 0.1509 | 0.5311 | -2.2% |
| q3_k_s+imatrix_en+boost_tr_x16_s0 | 821 | 0.4377 | 0.1507 | 0.5343 | -2.6% |
| q3_k_s+imatrix_en+boost_tr_x64_s0 | 821 | 0.4401 | 0.1517 | 0.5340 | -2.1% |
| q3_k_s+imatrix_en+boost_rand_x16_s0 | 821 | 0.4525 | 0.1510 | 0.5311 | +0.7% |
| q3_k_s+imatrix_en+boost_rand_x16_s1 | 821 | 0.4463 | 0.1513 | 0.5346 | -0.7% |
| q3_k_s+imatrix_en+boost_en_x16_s0 | 821 | 0.4488 | 0.1510 | 0.5318 | -0.1% |
| q3_k_s+imatrix_en+boost_fi_x16_s0 | 821 | 0.4495 | 0.1507 | 0.5165 | +0.0% |
| q3_k_s+imatrix_tr050+boost_tr_x16_s0 | 821 | 0.4099 | 0.1535 | 0.5043 | -8.8% |
| q3_k_s+imatrix_tr+boost_tr_x16_s0 | 821 | 0.4058 | 0.1563 | 0.4970 | -9.7% |
| q2_k+imatrix_en | 718 | 1.0068 | 0.3429 | 1.0672 | +0.0% |
| q2_k+imatrix_tr | 718 | 0.9064 | 0.3489 | 1.0649 | -10.0% |
| q2_k+imatrix_tr050 | 718 | 0.9043 | 0.3406 | 1.0281 | -10.2% |
| q2_k+imatrix_en+boost_tr_x4_s0 | 718 | 0.9951 | 0.3431 | 1.0645 | -1.2% |
| q2_k+imatrix_en+boost_tr_x16_s0 | 718 | 0.9928 | 0.3430 | 1.0699 | -1.4% |
| q2_k+imatrix_en+boost_tr_x64_s0 | 718 | 0.9910 | 0.3432 | 1.0681 | -1.6% |
| q2_k+imatrix_en+boost_rand_x16_s0 | 718 | 1.0084 | 0.3426 | 1.0670 | +0.2% |
| q2_k+imatrix_en+boost_rand_x16_s1 | 718 | 1.0079 | 0.3442 | 1.0706 | +0.1% |
| q2_k+imatrix_en+boost_en_x16_s0 | 718 | 1.0069 | 0.3435 | 1.0669 | +0.0% |
| q2_k+imatrix_en+boost_fi_x16_s0 | 718 | 1.0037 | 0.3426 | 1.0563 | -0.3% |
| q2_k+imatrix_tr050+boost_tr_x16_s0 | 718 | 0.9073 | 0.3410 | 1.0361 | -9.9% |
| q2_k+imatrix_tr+boost_tr_x16_s0 | 718 | 0.9103 | 0.3512 | 1.0670 | -9.6% |

### qwen2.5-1.5b: Fisher kestirimi ile ölçülen hassasiyet (Y2)

| Dil | Spearman ρ (kestirim vs ölçüm) |
| --- | --- |
| en | 0.899 |
| fi | 0.949 |
| tr | 0.916 |

TR/EN oranının kestirimi ile ölçümü arasında ρ = **0.924** (dile özgü hassasiyet farkını da yakalıyor mu?).

### qwen2.5-1.5b: Kalibrasyonda Türkçe payı (Y3)

trPPP = imatrix metninde %PPP Türkçe, gerisi İngilizce.

| Model | MB | KLD tr | KLD en | KLD fi | TR değişim (ref.) |
| --- | --- | --- | --- | --- | --- |
| q3_k_s+tr000 | 821 | 0.4491 | 0.1508 | 0.5328 | +0.0% |
| q2_k+tr000 | 718 | 1.0020 | 0.3435 | 1.0524 | +0.0% |
| q3_k_s+tr025 | 821 | 0.4227 | 0.1512 | 0.5094 | -5.9% |
| q2_k+tr025 | 718 | 0.9220 | 0.3443 | 1.0426 | -8.0% |
| q3_k_s+tr050 | 821 | 0.4094 | 0.1522 | 0.5002 | -8.8% |
| q2_k+tr050 | 718 | 0.9043 | 0.3406 | 1.0281 | -9.8% |
| q3_k_s+tr075 | 821 | 0.4015 | 0.1535 | 0.4930 | -10.6% |
| q2_k+tr075 | 718 | 0.9051 | 0.3450 | 1.0285 | -9.7% |
| q3_k_s+tr100 | 821 | 0.4075 | 0.1539 | 0.4886 | -9.3% |
| q2_k+tr100 | 718 | 0.9064 | 0.3535 | 1.0512 | -9.5% |

### qwen2.5-1.5b: Kalibrasyon dili

Q3_K_S.

| Model | MB | KLD tr | KLD en | KLD fi | TR değişim (ref.) |
| --- | --- | --- | --- | --- | --- |
| q3_k_s+imatrix_none | 821 | 0.5244 | 0.1925 | 0.5964 | +16.7% |
| q3_k_s+imatrix_en | 821 | 0.4494 | 0.1509 | 0.5302 | +0.0% |
| q3_k_s+imatrix_tr | 821 | 0.4039 | 0.1539 | 0.4956 | -10.1% |
| q3_k_s+imatrix_mix | 821 | 0.4120 | 0.1515 | 0.4998 | -8.3% |

### qwen2.5-1.5b: Dile göre bit dağıtımı (eşit boyut)

Taban Q3_K_S, seçilen 8 grup Q5_K.

| Model | MB | KLD tr | KLD en | KLD fi | TR değişim (ref.) |
| --- | --- | --- | --- | --- | --- |
| q3_k_s | 821 | 0.5244 | 0.1925 | 0.5964 | +67.7% |
| q3_k_s+en_aware | 867 | 0.3478 | 0.1411 | 0.4286 | +11.3% |
| q3_k_s+tr_aware | 867 | 0.3336 | 0.1411 | 0.4276 | +6.7% |
| q3_k_s+tr_excess_aware | 867 | 0.3568 | 0.1498 | 0.4352 | +14.1% |
| q3_k_s+en_aware+imatrix_en | 867 | 0.3126 | 0.1144 | 0.3890 | +0.0% |
| q3_k_s+tr_aware+imatrix_tr | 867 | 0.2727 | 0.1218 | 0.3610 | -12.8% |
| q3_k_s+tr_excess_aware+imatrix_tr | 867 | 0.2744 | 0.1237 | 0.3520 | -12.2% |

### qwen2.5-3b: Kalibrasyon dili

Q3_K_S.

| Model | MB | KLD tr | KLD en | KLD fi | TR değişim (ref.) |
| --- | --- | --- | --- | --- | --- |
| q3_k_s+imatrix_none | 1514 | 0.3765 | 0.1480 | 0.4920 | +17.3% |
| q3_k_s+imatrix_en | 1514 | 0.3211 | 0.1266 | 0.4386 | +0.0% |
| q3_k_s+imatrix_tr | 1514 | 0.2906 | 0.1324 | 0.4242 | -9.5% |
| q3_k_s+imatrix_mix | 1514 | 0.2925 | 0.1272 | 0.4260 | -8.9% |
