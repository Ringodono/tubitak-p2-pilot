# P2 pilot 3 raporu: Türkçe nöronları yüksek hassasiyette ayırma

Tüm model kuantize; seçilen FFN nöronları (gate/up satırı, down sütunu) F16'da tutulur. K = Türkçe LAPE nöron sayısı; xM = M*K nöron. err_en = dil bilgisi kullanmayan standart aktivasyon ağırlıklı hata seçimi; hyb = TR nöronları + kalan bütçe err_en. +MB, nöronları ayrı tabloda saklamanın ek boyutudur.

## qwen2.5-1.5b

### q3_k_s_en

| Seçim | Nöron | +MB | Toplam MB | KLD tr (değişim) | KLD en (değişim) |
| --- | --- | --- | --- | --- | --- |
| none | 0 | 0.0 | 821.3 | 0.4154 | 0.1408 |
| lape_tr | 954 | 6.6 | 827.9 | 0.3854 (-7.2%) | 0.1401 (-0.5%) |
| lape_tr_q8 | 954 | 2.7 | 824.0 | 0.3854 (-7.2%) | 0.1401 (-0.5%) |
| rand0 | 954 | 6.6 | 827.9 | 0.4132 (-0.5%) | 0.1403 (-0.4%) |
| rand1 | 954 | 6.6 | 827.9 | 0.4125 (-0.7%) | 0.1401 (-0.5%) |
| rand2 | 954 | 6.6 | 827.9 | 0.4142 (-0.3%) | 0.1400 (-0.6%) |
| imp_en | 954 | 6.6 | 827.9 | 0.4022 (-3.2%) | 0.1384 (-1.7%) |
| imp_tr | 954 | 6.6 | 827.9 | 0.3699 (-11.0%) | 0.1382 (-1.9%) |
| err_en_x0.5 | 477 | 3.3 | 824.6 | 0.4080 (-1.8%) | 0.1391 (-1.2%) |
| err_en_x1 | 954 | 6.6 | 827.9 | 0.4020 (-3.2%) | 0.1383 (-1.8%) |
| err_en_x2 | 1908 | 13.2 | 834.5 | 0.3963 (-4.6%) | 0.1358 (-3.6%) |
| err_en_x4 | 3816 | 26.4 | 847.7 | 0.3869 (-6.9%) | 0.1327 (-5.8%) |
| err_tr_x0.5 | 477 | 3.3 | 824.6 | 0.3810 (-8.3%) | 0.1394 (-1.0%) |
| err_tr_x1 | 954 | 6.6 | 827.9 | 0.3684 (-11.3%) | 0.1382 (-1.8%) |
| err_tr_x2 | 1908 | 13.2 | 834.5 | 0.3553 (-14.5%) | 0.1370 (-2.7%) |
| err_tr_x4 | 3816 | 26.4 | 847.7 | 0.3394 (-18.3%) | 0.1346 (-4.4%) |
| hyb_x2 | 1908 | 13.2 | 834.5 | 0.3748 (-9.8%) | 0.1375 (-2.4%) |
| hyb_x4 | 3816 | 26.4 | 847.7 | 0.3651 (-12.1%) | 0.1339 (-4.9%) |

### q3_k_s_tr

| Seçim | Nöron | +MB | Toplam MB | KLD tr (değişim) | KLD en (değişim) |
| --- | --- | --- | --- | --- | --- |
| none | 0 | 0.0 | 821.3 | 0.3733 | 0.1445 |
| lape_tr | 954 | 6.6 | 827.9 | 0.3625 (-2.9%) | 0.1440 (-0.4%) |
| lape_tr_q8 | 954 | 2.7 | 824.0 | 0.3624 (-2.9%) | 0.1440 (-0.4%) |
| rand0 | 954 | 6.6 | 827.9 | 0.3708 (-0.7%) | 0.1440 (-0.4%) |
| rand1 | 954 | 6.6 | 827.9 | 0.3713 (-0.5%) | 0.1440 (-0.4%) |
| rand2 | 954 | 6.6 | 827.9 | 0.3716 (-0.4%) | 0.1438 (-0.5%) |
| imp_en | 954 | 6.6 | 827.9 | 0.3643 (-2.4%) | 0.1419 (-1.9%) |
| imp_tr | 954 | 6.6 | 827.9 | 0.3507 (-6.0%) | 0.1425 (-1.4%) |
| err_en_x0.5 | 477 | 3.3 | 824.6 | 0.3681 (-1.4%) | 0.1426 (-1.3%) |
| err_en_x1 | 954 | 6.6 | 827.9 | 0.3649 (-2.3%) | 0.1419 (-1.8%) |
| err_en_x2 | 1908 | 13.2 | 834.5 | 0.3591 (-3.8%) | 0.1400 (-3.1%) |
| err_en_x4 | 3816 | 26.4 | 847.7 | 0.3536 (-5.3%) | 0.1370 (-5.2%) |
| err_tr_x0.5 | 477 | 3.3 | 824.6 | 0.3567 (-4.5%) | 0.1435 (-0.7%) |
| err_tr_x1 | 954 | 6.6 | 827.9 | 0.3496 (-6.4%) | 0.1426 (-1.3%) |
| err_tr_x2 | 1908 | 13.2 | 834.5 | 0.3373 (-9.7%) | 0.1413 (-2.2%) |
| err_tr_x4 | 3816 | 26.4 | 847.7 | 0.3256 (-12.8%) | 0.1390 (-3.9%) |
| hyb_x2 | 1908 | 13.2 | 834.5 | 0.3531 (-5.4%) | 0.1414 (-2.2%) |
| hyb_x4 | 3816 | 26.4 | 847.7 | 0.3437 (-7.9%) | 0.1379 (-4.6%) |

### q2_k_en

| Seçim | Nöron | +MB | Toplam MB | KLD tr (değişim) | KLD en (değişim) |
| --- | --- | --- | --- | --- | --- |
| none | 0 | 0.0 | 718.0 | 0.8480 | 0.2952 |
| lape_tr | 954 | 6.9 | 724.9 | 0.7931 (-6.5%) | 0.2929 (-0.8%) |
| rand0 | 954 | 6.9 | 724.9 | 0.8427 (-0.6%) | 0.2922 (-1.0%) |
| imp_tr | 954 | 6.9 | 724.9 | 0.7596 (-10.4%) | 0.2874 (-2.6%) |
| err_en_x1 | 954 | 6.9 | 724.9 | 0.8241 (-2.8%) | 0.2866 (-2.9%) |
| err_tr_x1 | 954 | 6.9 | 724.9 | 0.7590 (-10.5%) | 0.2873 (-2.7%) |
| err_en_x2 | 1908 | 13.7 | 731.7 | 0.8146 (-3.9%) | 0.2827 (-4.2%) |
| hyb_x2 | 1908 | 13.7 | 731.7 | 0.7720 (-9.0%) | 0.2846 (-3.6%) |

### q2_k_tr

| Seçim | Nöron | +MB | Toplam MB | KLD tr (değişim) | KLD en (değişim) |
| --- | --- | --- | --- | --- | --- |
| none | 0 | 0.0 | 718.0 | 0.7513 | 0.3026 |
| lape_tr | 954 | 6.9 | 724.9 | 0.7170 (-4.6%) | 0.3001 (-0.8%) |
| rand0 | 954 | 6.9 | 724.9 | 0.7456 (-0.8%) | 0.3006 (-0.7%) |
| imp_tr | 954 | 6.9 | 724.9 | 0.7052 (-6.1%) | 0.2952 (-2.5%) |
| err_en_x1 | 954 | 6.9 | 724.9 | 0.7303 (-2.8%) | 0.2919 (-3.6%) |
| err_tr_x1 | 954 | 6.9 | 724.9 | 0.7034 (-6.4%) | 0.2955 (-2.3%) |
| err_en_x2 | 1908 | 13.7 | 731.7 | 0.7221 (-3.9%) | 0.2878 (-4.9%) |
| hyb_x2 | 1908 | 13.7 | 731.7 | 0.6942 (-7.6%) | 0.2896 (-4.3%) |

### Boyut–kalite: nöron ayırma vs katman yükseltme (Q3_K_S)

Katman yükseltme: Türkçe profilde en hassas FFN katman(lar)ı daha yüksek türe çıkarılır.

| İmatrix | Yöntem | Toplam MB | KLD tr | KLD en | TR değişim |
| --- | --- | --- | --- | --- | --- |
| en | none | 821.3 | 0.4154 | 0.1408 | +0.0% |
| en | lape_tr_q8 | 824.0 | 0.3854 | 0.1401 | -7.2% |
| en | err_en_x0.5 | 824.6 | 0.4080 | 0.1391 | -1.8% |
| en | err_tr_x0.5 | 824.6 | 0.3810 | 0.1394 | -8.3% |
| en | up_q4_K_top1 | 826.6 | 0.4028 | 0.1368 | -3.0% |
| en | lape_tr | 827.9 | 0.3854 | 0.1401 | -7.2% |
| en | rand0 | 827.9 | 0.4132 | 0.1403 | -0.5% |
| en | rand1 | 827.9 | 0.4125 | 0.1401 | -0.7% |
| en | rand2 | 827.9 | 0.4142 | 0.1400 | -0.3% |
| en | imp_en | 827.9 | 0.4022 | 0.1384 | -3.2% |
| en | imp_tr | 827.9 | 0.3699 | 0.1382 | -11.0% |
| en | err_en_x1 | 827.9 | 0.4020 | 0.1383 | -3.2% |
| en | err_tr_x1 | 827.9 | 0.3684 | 0.1382 | -11.3% |
| en | up_q5_K_top1 | 831.5 | 0.3989 | 0.1371 | -4.0% |
| en | err_en_x2 | 834.5 | 0.3963 | 0.1358 | -4.6% |
| en | err_tr_x2 | 834.5 | 0.3553 | 0.1370 | -14.5% |
| en | hyb_x2 | 834.5 | 0.3748 | 0.1375 | -9.8% |
| en | up_q5_K_top2 | 841.6 | 0.3661 | 0.1294 | -11.9% |
| en | err_en_x4 | 847.7 | 0.3869 | 0.1327 | -6.9% |
| en | err_tr_x4 | 847.7 | 0.3394 | 0.1346 | -18.3% |
| en | hyb_x4 | 847.7 | 0.3651 | 0.1339 | -12.1% |

| tr | none | 821.3 | 0.3733 | 0.1445 | +0.0% |
| tr | lape_tr_q8 | 824.0 | 0.3624 | 0.1440 | -2.9% |
| tr | err_en_x0.5 | 824.6 | 0.3681 | 0.1426 | -1.4% |
| tr | err_tr_x0.5 | 824.6 | 0.3567 | 0.1435 | -4.5% |
| tr | up_q4_K_top1 | 826.6 | 0.3561 | 0.1392 | -4.6% |
| tr | lape_tr | 827.9 | 0.3625 | 0.1440 | -2.9% |
| tr | rand0 | 827.9 | 0.3708 | 0.1440 | -0.7% |
| tr | rand1 | 827.9 | 0.3713 | 0.1440 | -0.5% |
| tr | rand2 | 827.9 | 0.3716 | 0.1438 | -0.4% |
| tr | imp_en | 827.9 | 0.3643 | 0.1419 | -2.4% |
| tr | imp_tr | 827.9 | 0.3507 | 0.1425 | -6.0% |
| tr | err_en_x1 | 827.9 | 0.3649 | 0.1419 | -2.3% |
| tr | err_tr_x1 | 827.9 | 0.3496 | 0.1426 | -6.4% |
| tr | up_q5_K_top1 | 831.5 | 0.3517 | 0.1383 | -5.8% |
| tr | err_en_x2 | 834.5 | 0.3591 | 0.1400 | -3.8% |
| tr | err_tr_x2 | 834.5 | 0.3373 | 0.1413 | -9.7% |
| tr | hyb_x2 | 834.5 | 0.3531 | 0.1414 | -5.4% |
| tr | up_q5_K_top2 | 841.6 | 0.3305 | 0.1322 | -11.5% |
| tr | err_en_x4 | 847.7 | 0.3536 | 0.1370 | -5.3% |
| tr | err_tr_x4 | 847.7 | 0.3256 | 0.1390 | -12.8% |
| tr | hyb_x4 | 847.7 | 0.3437 | 0.1379 | -7.9% |

### Eşleştirilmiş karşılaştırmalar (Türkçe token başına KLD farkı, %95 GA)

Negatif = ilk seçim daha iyi. GA sıfırı içermiyorsa fark anlamlı.

| Taban | Karşılaştırma | Fark | %95 GA |
| --- | --- | --- | --- |
| q3_k_s_en | TR nöronları vs aynı sayıda rastgele | -0.02787 | -0.03286 … -0.02333 |
| q3_k_s_en | TR nöronları vs TR aktivasyon katkısı (aynı sayı) | +0.01552 | +0.01184 … +0.01941 |
| q3_k_s_en | TR nöronları vs standart hata tabanlı seçim (aynı sayı) | -0.01661 | -0.02181 … -0.01197 |
| q3_k_s_en | TR nöronları vs TR hata tabanlı seçim (aynı sayı) | +0.01698 | +0.01319 … +0.02089 |
| q3_k_s_en | TR nöronları + standart vs yalnız standart (2K) | -0.02157 | -0.02689 … -0.01704 |
| q3_k_s_en | TR nöronları + standart vs TR hata tabanlı (2K) | +0.01950 | +0.01565 … +0.02364 |
| q3_k_s_en | TR nöronları Q8_0 vs F16 | -0.00001 | -0.00020 … +0.00018 |
| q3_k_s_tr | TR nöronları vs aynı sayıda rastgele | -0.00877 | -0.01231 … -0.00519 |
| q3_k_s_tr | TR nöronları vs TR aktivasyon katkısı (aynı sayı) | +0.01173 | +0.00859 … +0.01502 |
| q3_k_s_tr | TR nöronları vs standart hata tabanlı seçim (aynı sayı) | -0.00238 | -0.00608 … +0.00156 |
| q3_k_s_tr | TR nöronları vs TR hata tabanlı seçim (aynı sayı) | +0.01289 | +0.00964 … +0.01628 |
| q3_k_s_tr | TR nöronları + standart vs yalnız standart (2K) | -0.00604 | -0.00962 … -0.00232 |
| q3_k_s_tr | TR nöronları + standart vs TR hata tabanlı (2K) | +0.01582 | +0.01231 … +0.01952 |
| q3_k_s_tr | TR nöronları Q8_0 vs F16 | -0.00004 | -0.00021 … +0.00013 |
| q2_k_en | TR nöronları vs aynı sayıda rastgele | -0.04958 | -0.05808 … -0.04112 |
| q2_k_en | TR nöronları vs TR aktivasyon katkısı (aynı sayı) | +0.03344 | +0.02630 … +0.04072 |
| q2_k_en | TR nöronları vs standart hata tabanlı seçim (aynı sayı) | -0.03098 | -0.03995 … -0.02204 |
| q2_k_en | TR nöronları vs TR hata tabanlı seçim (aynı sayı) | +0.03408 | +0.02664 … +0.04141 |
| q2_k_en | TR nöronları + standart vs yalnız standart (2K) | -0.04253 | -0.05118 … -0.03381 |
| q2_k_tr | TR nöronları vs aynı sayıda rastgele | -0.02865 | -0.03632 … -0.02108 |
| q2_k_tr | TR nöronları vs TR aktivasyon katkısı (aynı sayı) | +0.01176 | +0.00539 … +0.01791 |
| q2_k_tr | TR nöronları vs standart hata tabanlı seçim (aynı sayı) | -0.01328 | -0.02150 … -0.00541 |
| q2_k_tr | TR nöronları vs TR hata tabanlı seçim (aynı sayı) | +0.01356 | +0.00713 … +0.01980 |
| q2_k_tr | TR nöronları + standart vs yalnız standart (2K) | -0.02798 | -0.03563 … -0.02024 |

## qwen2.5-3b

### q3_k_s_en

| Seçim | Nöron | +MB | Toplam MB | KLD tr (değişim) | KLD en (değişim) |
| --- | --- | --- | --- | --- | --- |
| none | 0 | 0.0 | 1514.5 | 0.2983 | 0.1254 |
| lape_tr | 1501 | 13.8 | 1528.3 | 0.2751 (-7.8%) | 0.1249 (-0.4%) |
| rand0 | 1501 | 13.8 | 1528.3 | 0.2975 (-0.3%) | 0.1243 (-0.8%) |
| imp_tr | 1501 | 13.8 | 1528.3 | 0.2597 (-12.9%) | 0.1228 (-2.0%) |
| err_en_x1 | 1501 | 13.8 | 1528.3 | 0.2885 (-3.3%) | 0.1220 (-2.7%) |
| err_tr_x1 | 1501 | 13.8 | 1528.3 | 0.2590 (-13.2%) | 0.1230 (-1.9%) |
| err_en_x2 | 3002 | 27.6 | 1542.1 | 0.2840 (-4.8%) | 0.1197 (-4.5%) |
| hyb_x2 | 3002 | 27.6 | 1542.1 | 0.2661 (-10.8%) | 0.1216 (-3.0%) |

### q3_k_s_tr

| Seçim | Nöron | +MB | Toplam MB | KLD tr (değişim) | KLD en (değişim) |
| --- | --- | --- | --- | --- | --- |
| none | 0 | 0.0 | 1514.5 | 0.2661 | 0.1311 |
| lape_tr | 1501 | 13.8 | 1528.3 | 0.2562 (-3.7%) | 0.1306 (-0.4%) |
| rand0 | 1501 | 13.8 | 1528.3 | 0.2645 (-0.6%) | 0.1301 (-0.8%) |
| imp_tr | 1501 | 13.8 | 1528.3 | 0.2476 (-7.0%) | 0.1283 (-2.1%) |
| err_en_x1 | 1501 | 13.8 | 1528.3 | 0.2600 (-2.3%) | 0.1274 (-2.8%) |
| err_tr_x1 | 1501 | 13.8 | 1528.3 | 0.2473 (-7.1%) | 0.1285 (-2.0%) |
| err_en_x2 | 3002 | 27.6 | 1542.1 | 0.2581 (-3.0%) | 0.1249 (-4.7%) |
| hyb_x2 | 3002 | 27.6 | 1542.1 | 0.2490 (-6.4%) | 0.1269 (-3.2%) |

### Eşleştirilmiş karşılaştırmalar (Türkçe token başına KLD farkı, %95 GA)

Negatif = ilk seçim daha iyi. GA sıfırı içermiyorsa fark anlamlı.

| Taban | Karşılaştırma | Fark | %95 GA |
| --- | --- | --- | --- |
| q3_k_s_en | TR nöronları vs aynı sayıda rastgele | -0.02239 | -0.02618 … -0.01887 |
| q3_k_s_en | TR nöronları vs TR aktivasyon katkısı (aynı sayı) | +0.01537 | +0.01201 … +0.01863 |
| q3_k_s_en | TR nöronları vs standart hata tabanlı seçim (aynı sayı) | -0.01336 | -0.01724 … -0.00962 |
| q3_k_s_en | TR nöronları vs TR hata tabanlı seçim (aynı sayı) | +0.01615 | +0.01276 … +0.01953 |
| q3_k_s_en | TR nöronları + standart vs yalnız standart (2K) | -0.01786 | -0.02169 … -0.01444 |
| q3_k_s_tr | TR nöronları vs aynı sayıda rastgele | -0.00821 | -0.01095 … -0.00551 |
| q3_k_s_tr | TR nöronları vs TR aktivasyon katkısı (aynı sayı) | +0.00862 | +0.00583 … +0.01154 |
| q3_k_s_tr | TR nöronları vs standart hata tabanlı seçim (aynı sayı) | -0.00380 | -0.00702 … -0.00048 |
| q3_k_s_tr | TR nöronları vs TR hata tabanlı seçim (aynı sayı) | +0.00893 | +0.00604 … +0.01192 |
| q3_k_s_tr | TR nöronları + standart vs yalnız standart (2K) | -0.00904 | -0.01211 … -0.00622 |
