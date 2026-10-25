import numpy as np
import pytest


@pytest.fixture
def green_on_soil():
    rng = np.random.default_rng(0)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :128] = (60, 140, 50)
    img[:, 128:] = (120, 95, 70)
    noise = rng.integers(-8, 9, size=img.shape)
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def heads_image():
    img = np.full((200, 300, 3), 30, dtype=np.uint8)
    boxes = np.array([[10, 10, 40, 50], [100, 60, 140, 110], [220, 120, 270, 180]], dtype=np.float32)
    for x1, y1, x2, y2 in boxes.astype(int):
        img[y1:y2, x1:x2] = (220, 200, 120)
    return img, boxes
