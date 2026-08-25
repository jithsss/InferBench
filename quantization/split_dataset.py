from pathlib import Path
import random
import shutil


SOURCE_DIR = Path(
    "quantization/calibration/images"
)

CALIBRATION_DIR = Path(
    "quantization/calibration/calibration_set"
)

EVALUATION_DIR = Path(
    "quantization/calibration/evaluation"
)

CALIBRATION_COUNT = 40
SEED = 42


def main() -> None:
    image_paths = sorted(
        SOURCE_DIR.glob("*.jpg")
    )

    if len(image_paths) < (
        CALIBRATION_COUNT + 1
    ):
        raise RuntimeError(
            f"Need at least "
            f"{CALIBRATION_COUNT + 1} JPG images, "
            f"but found {len(image_paths)}."
        )

    rng = random.Random(SEED)

    shuffled = image_paths.copy()
    rng.shuffle(shuffled)

    calibration_images = shuffled[
        :CALIBRATION_COUNT
    ]

    evaluation_images = shuffled[
        CALIBRATION_COUNT:
    ]

    # Keep evaluation small for this initial experiment.
    evaluation_images = evaluation_images[:10]

    CALIBRATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove old split files.
    for output_dir in [
        CALIBRATION_DIR,
        EVALUATION_DIR,
    ]:
        for image in output_dir.glob("*.jpg"):
            image.unlink()

    # Copy calibration images.
    for image in calibration_images:
        shutil.copy2(
            image,
            CALIBRATION_DIR / image.name,
        )

    # Copy evaluation images.
    for image in evaluation_images:
        shutil.copy2(
            image,
            EVALUATION_DIR / image.name,
        )

    print(
        f"Calibration images: "
        f"{len(calibration_images)}"
    )

    print(
        f"Evaluation images: "
        f"{len(evaluation_images)}"
    )

    print("\nCalibration set:")
    for image in calibration_images:
        print(f"  {image.name}")

    print("\nEvaluation set:")
    for image in evaluation_images:
        print(f"  {image.name}")


if __name__ == "__main__":
    main()