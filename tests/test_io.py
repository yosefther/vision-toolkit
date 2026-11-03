import cv2
import numpy as np
import pytest

from vision_toolkit.io import describe, load_image

# One pixel per channel colour, in OpenCV's BGR order: pure blue, pure red.
BGR_PIXELS = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)


@pytest.fixture
def color_png(tmp_path):
    path = tmp_path / "color.png"
    cv2.imwrite(str(path), BGR_PIXELS)
    return path


@pytest.fixture
def alpha_png(tmp_path):
    path = tmp_path / "alpha.png"
    bgra = np.array([[[255, 0, 0, 77], [0, 0, 255, 200]]], dtype=np.uint8)
    cv2.imwrite(str(path), bgra)
    return path


# ---------- load_image ----------


def test_load_color_returns_bgr(color_png):
    img = load_image(color_png)
    assert img.shape == (1, 2, 3)
    assert img.dtype == np.uint8
    assert np.array_equal(img, BGR_PIXELS)


def test_load_accepts_str_path(color_png):
    img = load_image(str(color_png))
    assert np.array_equal(img, BGR_PIXELS)


def test_load_rgb_swaps_red_and_blue(color_png):
    img = load_image(color_png, rgb=True)
    assert np.array_equal(img, BGR_PIXELS[..., ::-1])


def test_load_grayscale_is_2d(color_png):
    img = load_image(color_png, mode="grayscale")
    assert img.ndim == 2
    assert img.shape == (1, 2)


def test_load_grayscale_ignores_rgb_flag(color_png):
    gray = load_image(color_png, mode="grayscale")
    gray_rgb = load_image(color_png, mode="grayscale", rgb=True)
    assert np.array_equal(gray, gray_rgb)


def test_load_color_drops_alpha(alpha_png):
    img = load_image(alpha_png)
    assert img.shape == (1, 2, 3)


def test_load_unchanged_keeps_alpha(alpha_png):
    img = load_image(alpha_png, mode="unchanged")
    assert img.shape == (1, 2, 4)
    assert img[0, 0, 3] == 77
    assert img[0, 1, 3] == 200


def test_load_unchanged_rgb_gives_rgba(alpha_png):
    img = load_image(alpha_png, mode="unchanged", rgb=True)
    # blue pixel -> (0, 0, 255) in RGB, alpha stays last
    assert img[0, 0].tolist() == [0, 0, 255, 77]
    assert img[0, 1].tolist() == [255, 0, 0, 200]


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.jpg")


def test_load_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path)


def test_load_corrupt_file_raises(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"this is not an image")
    with pytest.raises(ValueError):
        load_image(path)


# ---------- describe ----------


def test_describe_color_image():
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    img[0, 0] = [10, 20, 30]
    img[3, 5] = [200, 0, 0]
    assert describe(img) == {
        "width": 6,
        "height": 4,
        "channels": 3,
        "dtype": "uint8",
        "min": 0,
        "max": 200,
    }


def test_describe_grayscale_has_one_channel():
    img = np.full((3, 5), 42, dtype=np.uint8)
    info = describe(img)
    assert info["channels"] == 1
    assert (info["width"], info["height"]) == (5, 3)
    assert info["min"] == info["max"] == 42


def test_describe_alpha_has_four_channels():
    img = np.zeros((2, 2, 4), dtype=np.uint8)
    assert describe(img)["channels"] == 4


def test_describe_returns_plain_python_numbers():
    info = describe(np.full((2, 2), 7, dtype=np.uint16))
    assert info["dtype"] == "uint16"
    assert type(info["min"]) is int
    assert type(info["max"]) is int
