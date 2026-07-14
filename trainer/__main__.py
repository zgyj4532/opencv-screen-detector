import sys


def main() -> None:
    usage = "Usage: python -m trainer <train|export|train_pahvit|benchmark|validate_pahvit|ablation>"
    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    match sys.argv[1].lower():
        case "train":
            from .train import main as train_main

            train_main()
        case "export":
            from .export_onnx import main as export_main

            export_main()
        case "train_pahvit":
            from .train_pahvit import main as train_pahvit_main

            train_pahvit_main()
        case "benchmark":
            from .run_benchmark import main as benchmark_main

            benchmark_main()
        case "validate_pahvit":
            from .validate_pahvit import main as validate_pahvit_main

            validate_pahvit_main()
        case "ablation":
            from .ablation import main as ablation_main

            ablation_main()
        case x:
            print(f"Unknown command: {x}")
            print(usage)
            sys.exit(1)


if __name__ == "__main__":
    main()
