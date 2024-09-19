import os

import numpy as np
import onnxruntime as ort

from image import Preprocessor
from tokenizer import Tokenizer


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Computes softmax values for each sets of scores in x.
    This ensures the output sums to 1 for each image (along axis 1).
    """
    exp_arr = np.exp(x)
    return exp_arr / np.sum(exp_arr, axis=1, keepdims=True)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise cosine similarities between two embedding arrays.
    """
    def normalize(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    return normalize(x) @ normalize(y).T


class Clip:
    """
    CLIP inference using ONNX, without torch or torchvision.
    """
    def __init__(self, model="ViT-B/32", batch_size=None):

        allowed_models = ["ViT-B/32", "RN50"]
        if model not in allowed_models:
            raise ValueError(f"`model` must be in {allowed_models}")

        if model == "ViT-B/32":
            self.embedding_size = 512
            IMAGE_MODEL_FILE = "clip_image_model_vitb32.onnx"
            TEXT_MODEL_FILE = "clip_text_model_vitb32.onnx"

        elif model == "RN50":
            self.embedding_size = 1024
            IMAGE_MODEL_FILE = "clip_image_model_rn50.onnx"
            TEXT_MODEL_FILE = "clip_text_model_rn50.onnx"

        self._batch_size = batch_size

        base_dir = os.path.dirname(os.path.abspath(__file__))
        ps = ort.get_available_providers()

        # Load image model
        image_path = os.path.join(base_dir, IMAGE_MODEL_FILE)
        self._preprocessor = Preprocessor()
        self.image_model = ort.InferenceSession(image_path, providers=ps)

        # Load text model
        text_path = os.path.join(base_dir, TEXT_MODEL_FILE)
        self._tokenizer = Tokenizer()
        self.text_model = ort.InferenceSession(text_path, providers=ps)

    def embed_image(self, image):
        return self.embed_images([image], with_batching=False)

    def embed_text(self, text):
        return self.embed_texts([text], with_batching=False)

    def embed_images(self, images, with_batching=True) -> np.ndarray:
        """
        Compute the embeddings for a list of images.
        > images: A list of images to run on.
                  Each image must be a 3-channel (RGB) image.
                  Preprocessing resizes images to size (224, 224).

        Returns an array of embeddings of shape (len(images), embedding_size).
        """
        if not images:
            return self._get_empty_embedding()

        if not with_batching or self._batch_size is None:
            # Preprocess images
            images = [self._preprocessor.encode_image(im) for im in images]
            batch = np.concatenate(images)
            return self.image_model.run(None, {"IMAGE": batch})[0]

        else:
            embeddings = []
            for batch in to_batches(images, self._batch_size):
                embeddings.append(
                    self.embed_images(batch, with_batching=False)
                )

            if not embeddings:
                return self._get_empty_embedding()

            return np.concatenate(embeddings)

    def embed_texts(self, texts, with_batching=True) -> np.ndarray:
        """
        Compute the embeddings for a list of texts.
        > texts: list of texts to run on. Each is at most 77 characters.

        Returns an array of embeddings of shape (len(texts), embedding_size).
        """
        if not with_batching or self._batch_size is None:
            text = self._tokenizer.encode_text(texts)
            if len(text) == 0:
                return self._get_empty_embedding()

            return self.text_model.run(None, {"TEXT": text})[0]

        else:
            embeddings = []
            for batch in to_batches(texts, self._batch_size):
                embeddings.append(
                    self.embed_texts(batch, with_batching=False)
                )

            if not embeddings:
                return self._get_empty_embedding()

            return np.concatenate(embeddings)

    def _get_empty_embedding(self):
        return np.empty((0, self.embedding_size), dtype=np.float32)


def to_batches(items, size):
    """
    Splits an iterable (e.g. a list) into batches of length `size`.
    Includes the last, potentially shorter batch.
    """
    if size < 1:
        raise ValueError("Chunk size must be positive.")

    batch = []
    for item in items:
        batch.append(item)

        if len(batch) == size:
            yield batch
            batch = []

    if batch:
        yield batch
