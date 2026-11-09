# Kaynak ve Yapay Zekâ Beyanı

**Öğrenci:** Muhammed Emin Erdağ — **No:** 230212013

## Veri
- `data/normal.jpg`, `data/golge.jpg` ve `data/lamba.jpg` fotoğraflarını kendim çektim (Samsung Galaxy A54 5G, 9 Kasım 2026). Özgün dosya adları sırasıyla `20261109_200628.jpg`, `20261109_200638.jpg` ve `20261109_200618.jpg`'dir.
- Sahnede kişi yoktur. Başka bir kaynaktan veri kullanılmamıştır.

## Araç ve yapay zekâ kullanımı (ödev madde 8.e)
| | |
|---|---|
| **Kendi yaptıklarım** | Sahneyi kurdum; hedef nesneyi (turuncu kalemlik) ve üç ışık koşulunu seçtim; fotoğrafları çektim ve deneye uygunluklarını kontrol ettim; deneylerden önce tahminlerde bulundum, sonuçları ve raporu gözden geçirip tutarsız görünen yerleri düzelttim. |
| **Araç** | Claude (Anthropic), Claude Code ortamında |
| **Kullanım amacı** | Defterdeki kod hücrelerinin yazılması (yükleme, EXIF okuma, SIFT ile hizalama, maskeler, parlaklık/gama, küçültme, tablolar), rapor (`rapor/overleaf/main.tex` → PDF), `README.md` ve bu dosyanın taslağı |
| **Etkilediği dosya/hücre** | `odev1.ipynb`, `odev1.py`, `rapor/overleaf/main.tex`, `README.md`|
| **Nasıl doğrulandı** | Defter yeniden başlatılıp baştan sona çalıştırıldı; `odev1.py` de aynı `results/sonuc_tablosu.csv`'yi üretti. Rapordaki sayılar `results/` altındaki CSV/JSON dosyalarıyla karşılaştırıldı; maskeler ve küçültme şekilleri fotoğraflarla gözle kontrol edildi. |

Deney tahminleri, fotoğraflar çekilmeden ve deneyler çalıştırılmadan önce yazılmıştır. Ölçümlerden sonra değiştirilmemiştir; tutmayan tahminler raporda açıkça belirtilmiştir.

## Kaynaklar
1. YZM411 Bilgisayarlı Görü, Hafta 1–2 ders notu (görüntünün sayısal temsili, renk uzayları, örnekleme).
2. R. C. Gonzalez, R. E. Woods, *Digital Image Processing*, 4. baskı, Pearson, 2018. (yoğunluk dönüşümleri, gama, örnekleme ve örtüşme)
3. R. Szeliski, *Computer Vision: Algorithms and Applications*, 2. baskı, Springer, 2022. (görüntü oluşumu, öznitelik eşleme, homografi)
4. A. R. Smith, "Color gamut transform pairs," *SIGGRAPH Computer Graphics*, 12(3):12–19, 1978. (HSV renk uzayı)
5. D. G. Lowe, "Distinctive image features from scale-invariant keypoints," *International Journal of Computer Vision*, 60(2):91–110, 2004. (SIFT)
6. M. A. Fischler, R. C. Bolles, "Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography," *Communications of the ACM*, 24(6):381–395, 1981. (RANSAC)
7. N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1):62–66, 1979. (harf yüksekliği ölçümünde eşikleme)
8. OpenCV belgeleri, https://docs.opencv.org: `cvtColor`, `inRange`, `resize` (`INTER_NEAREST`, `INTER_AREA`), `SIFT_create`, `findHomography`, `warpPerspective`.
9. C. R. Harris vd., "Array programming with NumPy," *Nature*, 585:357–362, 2020.
