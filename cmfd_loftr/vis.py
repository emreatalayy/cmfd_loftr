# Visualization utilities for cmfd_loftr 

import torch

# ... (diğer kodlar)

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