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

class MaskPairDataset(Dataset):
    def __init__(self, root):
        self.imgs = []
        self.masks = []
        for fname in os.listdir(root):
            if fname.endswith('.png') or fname.endswith('.jpg'):
                img_path = os.path.join(root, fname)
                mask_path = os.path.splitext(img_path)[0] + '_mask.png'
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

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    ds = MaskPairDataset(args.dataset_root)
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    model = LoFTR(pretrained="outdoor").to(device)
    model.train()
    opt = Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for img, mask in pbar:
            img = img.to(device)
            mask = mask.to(device)
            batch = {"image0": img, "image1": img}
            model(batch)
            # LoFTR'ın matchability haritası: batch['mconf'] (N, Hc*Wc)
            if 'mconf' not in batch:
                continue
            mconf = batch['mconf'].view(-1)
            mask_flat = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(mconf.shape[0],), mode='nearest').squeeze(1).view(-1)
            loss = loss_fn(mconf, mask_flat)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix({'loss': float(loss)})
    print('Fine-tune tamamlandı!')

if __name__ == '__main__':
    main() 