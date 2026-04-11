import numpy as np

from src.vision_utils import TemporalSmoother, decode_prediction, prepare_classifier_input


def test_decode_prediction_uses_mask_class_index():
    label, confidence, is_mask = decode_prediction(np.array([0.2, 0.8], dtype=np.float32))

    assert label == "No Mask"
    assert is_mask is False
    assert np.isclose(confidence, 0.8)


def test_prepare_classifier_input_normalizes_uint8_image():
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    prepared = prepare_classifier_input(image)

    assert prepared.shape == (224, 224, 3)
    assert prepared.dtype == np.float32
    assert np.max(prepared) <= 1.0
    assert np.min(prepared) >= 0.0


def test_temporal_smoother_blends_probabilities():
    smoother = TemporalSmoother(alpha=0.5, max_distance=100, ttl=2)
    faces = [(0, 0, 20, 20, 0.9)]

    first = smoother.update(faces, [np.array([0.8, 0.2], dtype=np.float32)])
    second = smoother.update(faces, [np.array([0.2, 0.8], dtype=np.float32)])

    assert np.allclose(first[0], [0.8, 0.2], atol=1e-5)
    assert np.allclose(second[0], [0.5, 0.5], atol=1e-5)
