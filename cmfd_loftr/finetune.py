# Fine-tuning script for cmfd_loftr 

import argparse
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from kornia.feature import LoFTR
from torch.optim import Adam
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="LoFTR ile copy-move için fine-tune.")
    parser.add_argument('--dataset_root', required=True, help='Veri klasörü (görüntü+maske çiftleri)')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()

class Casia2MaskPairDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.imgs = []
        self.masks = []
        for fname in os.listdir(img_dir):
            if fname.endswith('.jpg') or fname.endswith('.tif'):
                img_path = os.path.join(img_dir, fname)
                mask_name = os.path.splitext(fname)[0] + '_gt.png'
                mask_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(mask_path):
                    self.imgs.append(img_path)
                    self.masks.append(mask_path)
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask)
        return img, mask

class LoFTRMaskNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loftr = LoFTR(pretrained="outdoor")
        # LoFTR backbone'dan çıkan feature boyutu genellikle 256'dır (coarse feature)
        self.mask_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, 1)
        )
    def forward(self, img):
        # img: [B, 1, H, W]
        # LoFTR backbone'dan feature çıkar
        feats_c, _ = self.loftr.backbone(img)  # feats_c: [B, 256, Hc, Wc]
        mask_pred = self.mask_head(feats_c)
        # mask_pred: [B, 1, Hc, Wc]
        return mask_pred

def compute_metrics(pred, gt, threshold=0.5):
    pred_bin = (pred > threshold).float()
    gt_bin = (gt > 0.5).float()
    intersection = (pred_bin * gt_bin).sum().item()
    union = (pred_bin + gt_bin - pred_bin * gt_bin).sum().item()
    iou = intersection / (union + 1e-8)
    tp = intersection
    fp = (pred_bin * (1 - gt_bin)).sum().item()
    fn = ((1 - pred_bin) * gt_bin).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {'iou': iou, 'precision': precision, 'recall': recall, 'f1': f1}

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"Using device: {device}")
    ds = Casia2MaskPairDataset(os.path.join(args.dataset_root, 'Tp'), os.path.join(args.dataset_root, 'CASIA 2 Groundtruth'))
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    model = LoFTRMaskNet().to(device)
    model.train()
    opt = Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        model.train()
        for img, mask in pbar:
            img = img.to(device)
            mask = mask.to(device)
            mask_pred = model(img)
            mask_resized = torch.nn.functional.interpolate(mask.unsqueeze(1), size=mask_pred.shape[2:], mode='nearest')
            loss = loss_fn(mask_pred, mask_resized)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix({'loss': float(loss)})
        # --- Validation & Metrics ---
        model.eval()
        iou_total, f1_total, prec_total, rec_total, n = 0, 0, 0, 0, 0
        with torch.no_grad():
            for img, mask in loader:
                img = img.to(device)
                mask = mask.to(device)
                mask_pred = model(img)
                mask_resized = torch.nn.functional.interpolate(mask.unsqueeze(1), size=mask_pred.shape[2:], mode='nearest')
                metrics = compute_metrics(torch.sigmoid(mask_pred), mask_resized)
                iou_total += metrics['iou']
                f1_total += metrics['f1']
                prec_total += metrics['precision']
                rec_total += metrics['recall']
                n += 1
        print(f"[Validation] IoU: {iou_total/n:.4f}, F1: {f1_total/n:.4f}, Precision: {prec_total/n:.4f}, Recall: {rec_total/n:.4f}")
    # --- Modeli Kaydet ---
    torch.save(model.state_dict(), "loftr_masknet_casia2.pth")
    print("Model ağırlıkları 'loftr_masknet_casia2.pth' olarak kaydedildi.")

if __name__ == '__main__':
    main() 