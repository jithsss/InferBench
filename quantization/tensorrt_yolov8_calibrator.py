import tensorrt as trt
import torch
import numpy as np
from PIL import Image
from pathlib import Path

class YoloEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_dir: str, input_shape: tuple[int, int, int, int], cache_file: str, max_images: int = 40):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.max_images = max_images
        
        self.image_paths = sorted(self.image_dir.glob("*.jpg")) + sorted(self.image_dir.glob("*.png"))
        self.image_paths = self.image_paths[:self.max_images]
        self.current_index = 0
        self.device_input = None

        if not self.image_paths:
            raise RuntimeError(f"No JPG/PNG images found in {image_dir}")

        if input_shape[0] != 1:
            raise ValueError("This calibrator expects batch size 1.")

    def preprocess(self, image_path: Path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.input_shape[3], self.input_shape[2]))
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        # YOLOv8 standard preprocessing (CHW, normalize 0-1)
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        return np.ascontiguousarray(image_array, dtype=np.float32)

    def get_batch_size(self):
        return self.input_shape[0]

    def get_batch(self, names):
        if self.current_index >= len(self.image_paths):
            return None

        image_path = self.image_paths[self.current_index]
        self.current_index += 1

        batch = self.preprocess(image_path)
        self.device_input = torch.from_numpy(batch).cuda()
        
        return [int(self.device_input.data_ptr())]

    def read_calibration_cache(self):
        cache_path = Path(self.cache_file)
        if cache_path.exists():
            print(f"Using calibration cache: {cache_path}")
            return cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        cache_path = Path(self.cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(cache)
        print(f"Calibration cache saved to: {cache_path}")
