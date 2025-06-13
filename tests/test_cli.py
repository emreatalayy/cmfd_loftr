# Unit tests for cmfd_loftr CLI 
import os
import numpy as np
import cv2
import tempfile
import subprocess
import sys

def test_cli_round_trip():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.png")
        mask_path = os.path.join(tmpdir, "mask.png")
        cv2.imwrite(img_path, img)
        cmd = [sys.executable, "-m", "cmfd_loftr.cli", "--input", img_path, "--output", mask_path, "--device", "cpu"]
        result = subprocess.run(cmd, capture_output=True)
        assert result.returncode == 0
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        assert mask.shape[:2] == (64, 64)
        assert mask.dtype == np.uint8 