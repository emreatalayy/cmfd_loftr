# Unit tests for cmfd_loftr.detect.detect_copy_move 
import numpy as np
import pytest
from cmfd_loftr.detect import detect_copy_move

def test_output_shape_and_dtype():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    mask = detect_copy_move(img, device="cpu")
    assert mask.shape == (128, 128)
    assert mask.dtype == np.uint8
    assert np.all((mask == 0) | (mask == 1))

def test_dbscan_removes_singletons():
    # Create a fake image with random noise, expect no clusters
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    mask = detect_copy_move(img, device="cpu", min_samples=10)
    # Should be all zeros (no region large enough)
    assert np.count_nonzero(mask) == 0

def test_deterministic_output():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    mask1 = detect_copy_move(img, device="cpu")
    mask2 = detect_copy_move(img, device="cpu")
    assert np.array_equal(mask1, mask2) 