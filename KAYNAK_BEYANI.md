# Kaynak ve Yapay Zekâ Beyanı

## Veri
- `data/normal.jpg`, `data/golge.jpg` ve `data/lamba.jpg` fotoğraflarını kendim çektim (Samsung Galaxy A54 5G, 9 Kasım 2026). Özgün dosya adları sırasıyla `20261109_200628.jpg`, `20261109_200638.jpg` ve `20261109_200618.jpg`'dir.
- Sahnede kişi yoktur. Başka bir kaynaktan veri kullanılmamıştır.

## Yapay zekâ kullanımı
| | |
|---|---|
| **Araç** | Claude (Anthropic), Claude Code ortamında |
| **Kullanım amacı** | Defterdeki tüm kod hücrelerinin yazılması (yükleme, EXIF okuma, SIFT ile hizalama, maskeler, parlaklık/gama, küçültme, tablolar), deney yorumlarının ilk taslağı, eşik türetme yönteminin önerilmesi, rapor (`rapor/rapor.html` → PDF), `README.md` ve bu dosyanın taslağı |
| **Etkilediği dosya/hücre** | `odev1.ipynb` (tüm kod ve yorum hücreleri), `odev1.py`, `rapor/rapor.html`, `rapor/YZM411_Odev1_Rapor.pdf`, `README.md`, `KAYNAK_BEYANI.md` |
| **Nasıl doğruladım** | Defteri yeniden başlatıp baştan sona çalıştırdım; `odev1.py` de aynı `results/sonuc_tablosu.csv`'yi üretti. Rapordaki sayıları `results/` altındaki CSV/JSON dosyalarıyla karşılaştırdım; maskeleri ve küçültme şekillerini fotoğraflarla gözle kontrol ettim. |

Deney tahminleri, fotoğraflar çekilmeden ve deneyler çalıştırılmadan önce yazılmıştır. Ölçümlerden sonra değiştirilmemiştir; tutmayan tahminler raporda açıkça belirtilmiştir.

## Kaynaklar
- YZM411 Bilgisayarlı Görü, Hafta 1–2 ders notları (görüntünün sayısal temsili, renk uzayları, örnekleme)
- OpenCV belgeleri: `cvtColor`, `inRange`, `resize` (`INTER_NEAREST`, `INTER_AREA`), `SIFT_create`, `findHomography`, `warpPerspective`
- NumPy belgeleri
