from pathlib import Path

from PIL import Image
from torchvision.datasets import CIFAR10


OUTPUT_DIR = Path("quantization/calibration/images")
NUM_IMAGES = 50


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading CIFAR-10...")

    dataset = CIFAR10(
        root="quantization/calibration/data",
        train=False,
        download=True,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Preparing {NUM_IMAGES} calibration images...")

    for index in range(NUM_IMAGES):
        image, label = dataset[index]

        # CIFAR-10 returns a PIL image.
        image = image.convert("RGB")

        output_path = OUTPUT_DIR / f"image_{index:03d}.jpg"

        image.save(
            output_path,
            format="JPEG",
            quality=95,
        )

        print(
            f"Saved {output_path} "
            f"(class={label})"
        )

    print(
        f"\nCalibration images saved to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()