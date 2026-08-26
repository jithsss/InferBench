import onnxruntime_genai as og


MODEL_PATH = "models/qwen3-0.6b"


def main() -> None:
    print("Loading Qwen3-0.6B configuration...")

    config = og.Config(MODEL_PATH)

    config.clear_providers()
    config.append_provider("cuda")

    print("Execution provider: CUDA")

    model = og.Model(config)

    print("Qwen3-0.6B loaded successfully.")


if __name__ == "__main__":
    main()