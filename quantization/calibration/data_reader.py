from pathlib import Path

import numpy as np
from PIL import Image
from onnxruntime.quantization import CalibrationDataReader


class ResNetCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        image_dir: str,
        input_name: str,
        max_images: int = 50,
    ) -> None:
        self.input_name = input_name
        self.image_paths = list(Path(image_dir).glob("*.jpg"))[:max_images]
        self.index = 0

    @staticmethod
    def preprocess(image_path: Path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))

        image_array = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

        # ImageNet normalization used by pretrained ResNet50.
        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

        image_array = (image_array - mean) / std

        # HWC -> CHW
        image_array = np.transpose(
            image_array,
            (2, 0, 1),
        )

        # Add batch dimension.
        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        return image_array

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        image_path = self.image_paths[self.index]
        self.index += 1

        return {
            self.input_name: self.preprocess(image_path)
        }

    def rewind(self) -> None:
        self.index = 0