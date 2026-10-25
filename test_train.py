import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


@pytest.mark.slow
def test_training_smoke(heads_image):
    from wheatkit.tasks import count_heads, torchvision_detector
    from wheatkit.train.detector import TrainConfig, build_model, train_detector

    img, boxes = heads_image
    model = build_model("fasterrcnn_mobilenet_v3_large_fpn", pretrained_backbone=False)
    model.transform.min_size, model.transform.max_size = (200,), 320
    cfg = TrainConfig(epochs=1, batch_size=2, max_steps=1)
    trained = train_detector([img, img], [boxes, boxes], cfg, model=model)
    det = torchvision_detector(trained, score_threshold=0.0)
    r = count_heads(img, det)
    assert isinstance(r.count, int) and r.boxes.shape[1] == 4
    assert np.all(r.scores >= 0)
