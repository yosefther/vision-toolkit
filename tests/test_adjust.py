import numpy as np
import pytest

from vision_toolkit.adjust import adjust_brightness, adjust_contrast


def make(values):
    return np.array([values], dtype=np.uint8)


# ---------- adjust_brightness ----------


def test_brightness_adds_amount():
    out = adjust_brightness(make([0, 100, 200]), 40)
    assert np.array_equal(out, make([40, 140, 240]))


def test_brightness_negative_amount_darkens():
    out = adjust_brightness(make([50, 100, 200]), -30)
    assert np.array_equal(out, make([20, 70, 170]))


def test_brightness_clips_at_255():
    out = adjust_brightness(make([250]), 10)
    assert out[0, 0] == 255  # not 4 (no uint8 wraparound)


def test_brightness_clips_at_0():
    out = adjust_brightness(make([50]), -100)
    assert out[0, 0] == 0


def test_brightness_rounds_decimals():
    out = adjust_brightness(make([100]), 10.7)
    assert out[0, 0] == 111


def test_brightness_zero_amount_is_identity():
    img = make([0, 1, 128, 254, 255])
    assert np.array_equal(adjust_brightness(img, 0), img)


def test_brightness_large_amount_does_not_crash():
    out = adjust_brightness(make([0, 255]), 40000)
    assert np.array_equal(out, make([255, 255]))


# ---------- adjust_contrast ----------


def test_contrast_factor_one_is_identity():
    img = make([0, 50, 128, 200, 255])
    assert np.array_equal(adjust_contrast(img, 1), img)


@pytest.mark.parametrize("factor", [0, 0.5, 1, 1.7, 3])
def test_contrast_mid_gray_never_changes(factor):
    assert adjust_contrast(make([128]), factor)[0, 0] == 128


def test_contrast_factor_zero_gives_flat_gray():
    out = adjust_contrast(make([0, 64, 200, 255]), 0)
    assert np.array_equal(out, make([128, 128, 128, 128]))


def test_contrast_increases_distance_from_mid_gray():
    out = adjust_contrast(make([100, 160]), 2)
    assert np.array_equal(out, make([72, 192]))


def test_contrast_decreases_distance_from_mid_gray():
    out = adjust_contrast(make([28, 228]), 0.5)
    assert np.array_equal(out, make([78, 178]))


def test_contrast_clips_both_ends():
    out = adjust_contrast(make([10, 250]), 3)
    assert np.array_equal(out, make([0, 255]))


def test_contrast_rounds_decimals():
    # (100 - 128) * 0.9 + 128 = 102.8 -> 103
    assert adjust_contrast(make([100]), 0.9)[0, 0] == 103


# ---------- shared behaviour ----------


@pytest.mark.parametrize(
    "func, arg",
    [(adjust_brightness, 25), (adjust_contrast, 1.5)],
)
def test_keeps_dtype_and_shape(func, arg):
    img = np.random.default_rng(0).integers(0, 256, (5, 7, 3), dtype=np.uint8)
    out = func(img, arg)
    assert out.dtype == np.uint8
    assert out.shape == img.shape


@pytest.mark.parametrize(
    "func, arg",
    [(adjust_brightness, 25), (adjust_contrast, 1.5)],
)
def test_does_not_modify_input(func, arg):
    img = make([10, 100, 200])
    before = img.copy()
    func(img, arg)
    assert np.array_equal(img, before)


@pytest.mark.parametrize(
    "func, arg",
    [(adjust_brightness, 25), (adjust_contrast, 1.5)],
)
def test_works_on_grayscale(func, arg):
    img = np.full((4, 4), 100, dtype=np.uint8)
    out = func(img, arg)
    assert out.shape == (4, 4)
