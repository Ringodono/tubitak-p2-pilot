# P2 pilot 2 raporu

Modeller: v1 (results/)

## 1. Olgu: Türkçe kırılganlığı ve katman profili

| Model | TR/EN KLD (Q3_K_S, imatrix yok) | FI/EN | Profil ρ (TR–EN) | TR/EN ilk %20 katman | orta | son %20 |
| --- | --- | --- | --- | --- | --- | --- |
| v1 (results/) | 2.72 | - | 0.874 | 3.67 | 1.90 | 2.72 |

## 2. Mekanizma (M1–M3)

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
