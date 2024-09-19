from typing import Union

import numpy as np
from PIL import Image


class Preprocessor:
    """
    Our approach to the CLIP `preprocess` neural net that does not rely on PyTorch.
    The two approaches fully match.
    """
    CLIP_INPUT_SIZE = 224

    # Normalization constants taken from original CLIP:
    NORM_MEAN = np.array([0.48145466, 0.4578275, 0.40821073]).reshape((1, 1, 3))
    NORM_STD = np.array([0.26862954, 0.26130258, 0.27577711]).reshape((1, 1, 3))

    @staticmethod
    def _crop_and_resize(img: np.ndarray) -> np.ndarray:
        """
        Resize and crop an image to a square, preserving the aspect ratio.
        """
        h, w = img.shape[0:2]

        if h * w == 0:
            raise ValueError("Invalid image shape.")

        target_size = Preprocessor.CLIP_INPUT_SIZE

        # Resize so that the smaller dimension matches the required input size.
        if h < w:
            resized_h = target_size
            resized_w = int(resized_h * w / h)
        else:
            resized_w = target_size
            resized_h = int(resized_w * h / w)

        # We're working with float images but PIL uses uint8, so convert
        # there and back again afterwards
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil = img_pil.resize((resized_w, resized_h), resample=Image.BICUBIC)
        img = np.array(img_pil).astype(np.float32) / 255

        # Now crop to a square
        y_from = (resized_h - target_size) // 2
        x_from = (resized_w - target_size) // 2
        img = img[y_from: y_from + target_size, x_from: x_from + target_size]

        return img

    @staticmethod
    def _normalize(img: Union[Image.Image, np.ndarray]):
        """
        (H, W, 3), np.float32, between [0, 1], no NaNs.
        """
        if not isinstance(img, (Image.Image, np.ndarray)):
            raise TypeError("Expect PIL or np.array")

        if isinstance(img, Image.Image):
            img = np.array(img)

        if len(img.shape) > 3:
            raise ValueError("Expected 2 or 3 dimensions")

        if len(img.shape) == 3 and img.shape[2] != 3:
            raise ValueError("Expected 3-channel RGB image")

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
            img = np.concatenate((img,) * 3, axis=2)

        if np.min(img) < 0:
            raise ValueError(
                "Images should have non-negative pixel values, "
                f"but the minimum value is {np.min(img)}"
            )

        if np.issubdtype(img.dtype, np.floating):
            if np.max(img) > 1:
                raise ValueError("Float larger than 1")
            img = img.astype(np.float32)

        elif np.issubdtype(img.dtype, np.integer):
            if np.max(img) > 255:
                raise ValueError("Int larger than 255")
            img = img.astype(np.float32) / 255
            img = np.clip(img, 0, 1)  # In case of rounding errors

        else:
            raise ValueError(f"Unsupported dtype: {img.dtype}.")

        if np.isnan(img).any():
            raise ValueError("The image contains NaN values.")

        assert np.min(img) >= 0
        assert np.max(img) <= 1
        assert img.dtype == np.float32
        assert len(img.shape) == 3
        assert img.shape[2] == 3

        return img

    def encode_image(self, img: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocesses the images like CLIP's preprocess() function.
        """
        img = Preprocessor._normalize(img)
        img = Preprocessor._crop_and_resize(img)

        # Normalize channels
        img = (img - Preprocessor.NORM_MEAN) / Preprocessor.NORM_STD

        # Mimic the pytorch tensor format for Model class
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)

        return img
