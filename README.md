# cmfd_loftr: LoFTR ile Görüntüde Kopyala-Yapıştır Sahtecilik Tespiti

## Özellikler
- Tek bir görüntüde kopyala-yapıştır (copy-move) sahteciliği tespiti
- LoFTR (Kornia) ile self-matching
- Çok ölçekli analiz, DBSCAN ve RANSAC ile güvenilir bölge çıkarımı
- Komut satırı arayüzü (CLI) ve Python API
- Fine-tune (isteğe bağlı) destekli

## Kurulum

```bash
pip install -e .
```
Gerekli ek paketler otomatik kurulur. (torch, kornia, opencv-python, scikit-learn, tqdm, pillow, rich)

## Kullanım

### Python API
```python
import cv2
from cmfd_loftr.detect import detect_copy_move
img = cv2.imread('ornek.jpg')
mask = detect_copy_move(img, device='cuda')
cv2.imwrite('mask.png', (mask*255).astype('uint8'))
```

### Komut Satırı (CLI)
```bash
python -m cmfd_loftr.cli --input ornek.jpg --output mask.png --viz
```
- `--viz` ile maske üstüne bindirilmiş görsel de kaydedilir (mask.viz.png).
- `--device cpu` ile CPU'da çalıştırabilirsiniz.

## Fine-tune (İnce Ayar) Scripti
Kendi veri setinizle LoFTR'ı maske eşleşmesi için eğitmek isterseniz:
```bash
python -m cmfd_loftr.finetune --dataset_root VERI_KLASORU --epochs 5 --lr 1e-4
```
- Veri klasörü: Görüntü ve ground-truth maske çiftleri içermelidir.
- Kayıp fonksiyonu: Binary cross-entropy (BCE)

## CASIA2 ile Fine-tune
CASIA2 veri seti ile fine-tune yapmak için:

```bash
python -m cmfd_loftr.finetune --dataset_root CASIA2 --epochs 5 --lr 1e-4 --device cuda
```
- Görüntüler: `CASIA2/Tp/`
- Maskeler: `CASIA2/CASIA 2 Groundtruth/`
- Maskeler, görüntü dosya adının sonuna `_gt.png` eklenerek eşleşir.

## Testler
Tüm testleri çalıştırmak için:
```bash
pytest -q
```

## Lisans
MIT


