from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from PIL import Image


class ResNetEntropyCalibrator(
    trt.IInt8EntropyCalibrator2
):
    def __init__(
        self,
        image_dir: str,
        input_shape: tuple[int, int, int, int],
        cache_file: str,
        max_images: int = 40,
    ) -> None:
        super().__init__()

        self.batch_size = input_shape[0]
        self.input_shape = input_shape
        self.cache_file = cache_file

        self.image_paths = sorted(
            Path(image_dir).glob("*.jpg")
        )[:max_images]

        self.index = 0
        self.device_input = None

        if not self.image_paths:
            raise RuntimeError(
                f"No JPG images found in {image_dir}"
            )

        if self.batch_size != 1:
            raise ValueError(
                "This calibrator expects batch size 1."
            )

    @staticmethod
    def preprocess(
        image_path: Path,
    ) -> np.ndarray:
        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            (224, 224)
        )

        image_array = (
            np.asarray(
                image,
                dtype=np.float32,
            )
            / 255.0
        )

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

        image_array = (
            image_array - mean
        ) / std

        image_array = np.transpose(
            image_array,
            (2, 0, 1),
        )

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        return np.ascontiguousarray(
            image_array,
            dtype=np.float32,
        )

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names):
        if self.index >= len(
            self.image_paths
        ):
            return None

        image_path = self.image_paths[
            self.index
        ]

        batch = self.preprocess(
            image_path
        )

        self.index += 1

        self.device_input = (
            torch.from_numpy(batch).cuda()
        )

        return [
            int(
                self.device_input.data_ptr()
            )
        ]

    def read_calibration_cache(self):
        cache_path = Path(
            self.cache_file
        )

        if cache_path.exists():
            print(
                f"Using calibration cache: "
                f"{cache_path}"
            )

            return cache_path.read_bytes()

        return None

    def write_calibration_cache(
        self,
        cache,
    ):
        cache_path = Path(
            self.cache_file
        )

        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cache_path.write_bytes(
            cache
        )

        print(
            f"Calibration cache saved to: "
            f"{cache_path}"
        )