import numpy as np
import torch
import cv2
from typing import Tuple, Sequence
from sklearn.cluster import DBSCAN
from kornia.feature import LoFTR
import kornia as K
from rich.console import Console
import time


def detect_copy_move(
    image: np.ndarray,
    *,
    scales: Sequence[float] = (0.5, 1, 2),
    dbscan_eps: float = 3.0,
    min_samples: int = 20,
    ransac_thresh: float = 3.0,
    device: str = "cuda",
    verbose: bool = False
) -> np.ndarray:
    """
    Detect copy-move forgeries in a single image using LoFTR self-matching.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W, 3) or (H, W), dtype uint8 or float32.
    scales : Sequence[float], optional
        Image pyramid scales to process, by default (0.5, 1, 2).
    dbscan_eps : float, optional
        DBSCAN epsilon for clustering translation vectors, by default 3.0.
    min_samples : int, optional
        Minimum samples per DBSCAN cluster, by default 20.
    ransac_thresh : float, optional
        RANSAC affine threshold (pixels), by default 3.0.
    device : str, optional
        Torch device ("cuda" or "cpu"), by default "cuda".
    verbose : bool, optional
        If True, logs timing and progress with rich, by default False.

    Returns
    -------
    np.ndarray
        Binary mask of detected copy-move regions, shape (H, W), dtype uint8, 1=copied, 0=background.
    """
    console = Console() if verbose else None
    t0 = time.time()
    orig_h, orig_w = image.shape[:2]
    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

    # Prepare image: ensure float32, 0-1, 1xCxHxW
    if image.ndim == 2:
        img = image[..., None]
    else:
        img = image
    img = img.astype(np.float32)
    if img.max() > 1.1:
        img = img / 255.0
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(-1)
    masks = []
    all_mkpts0, all_mkpts1 = [], []
    all_scales = []

    if verbose:
        console.log(f"[bold cyan]Building image pyramid: scales={scales}")
    # Build image pyramid and run LoFTR self-matching
    for scale in scales:
        t1 = time.time()
        h_s, w_s = int(orig_h * scale), int(orig_w * scale)
        img_s = cv2.resize(img, (w_s, h_s), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(img_s).float().unsqueeze(0).unsqueeze(0).to(device)  # 1x1xHxW
        with torch.no_grad():
            if device == "cuda":
                with torch.cuda.amp.autocast():
                    matcher = LoFTR(pretrained="outdoor").to(device).eval()
                    batch = {"image0": img_t, "image1": img_t}
                    matcher(batch)
            else:
                matcher = LoFTR(pretrained="outdoor").to(device).eval()
                batch = {"image0": img_t, "image1": img_t}
                matcher(batch)
        if "mkpts0_f" not in batch or "mkpts1_f" not in batch:
            continue
        mkpts0 = batch["mkpts0_f"].cpu().numpy()  # (N,2)
        mkpts1 = batch["mkpts1_f"].cpu().numpy()  # (N,2)
        if mkpts0.shape[0] == 0:
            continue
        all_mkpts0.append(mkpts0 / scale)  # rescale to original
        all_mkpts1.append(mkpts1 / scale)
        all_scales.append(np.full((mkpts0.shape[0],), scale, dtype=np.float32))
        if verbose:
            console.log(f"[green]Scale {scale}: {mkpts0.shape[0]} matches in {time.time()-t1:.2f}s")

    if len(all_mkpts0) == 0:
        if verbose:
            console.log("[yellow]No matches found at any scale.")
        return np.zeros((orig_h, orig_w), dtype=np.uint8)

    mkpts0 = np.concatenate(all_mkpts0, axis=0)
    mkpts1 = np.concatenate(all_mkpts1, axis=0)
    translations = mkpts1 - mkpts0
    norms = np.linalg.norm(translations, axis=1)
    keep = norms >= 2.0
    mkpts0, mkpts1, translations = mkpts0[keep], mkpts1[keep], translations[keep]
    if verbose:
        console.log(f"[bold cyan]Total matches after filtering: {len(mkpts0)}")

    if len(mkpts0) == 0:
        return np.zeros((orig_h, orig_w), dtype=np.uint8)

    # DBSCAN clustering
    t2 = time.time()
    db = DBSCAN(eps=dbscan_eps, min_samples=min_samples)
    labels = db.fit_predict(translations)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if verbose:
        console.log(f"[bold cyan]DBSCAN found {n_clusters} clusters in {time.time()-t2:.2f}s")

    mask_total = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # noise
        idx = np.where(labels == cluster_id)[0]
        if len(idx) < min_samples:
            continue
        src = mkpts0[idx]
        dst = mkpts1[idx]
        # RANSAC affine
        if len(src) < 3:
            continue
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        if inliers is None:
            continue
        inliers = inliers.ravel().astype(bool)
        inlier_ratio = inliers.sum() / len(inliers)
        if inlier_ratio < 0.6:
            continue
        # Rasterise inlier keypoints
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for pt in src[inliers]:
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(mask, (x, y), 3, 1, -1)
        # Morph close
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        # Connected components filter
        n_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= 500:
                mask_total[labels_cc == i] = 1
    if verbose:
        console.log(f"[bold cyan]Total time: {time.time()-t0:.2f}s")
    return mask_total.astype(np.uint8) 