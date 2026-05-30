import torch
import logging
from argparse import Namespace
from simple_parsing import ArgumentParser
from configs import ModelConfig
from utils import config_logger, RGB_BASED_MODELS, POSE_BASED_MODELS
from tools import load_model
from tools.models import get_input_shape


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Convert a trained model to ONNX format",
    )
    parser.add_arguments(ModelConfig, "model")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for ONNX model. Defaults to '<arch>.onnx'",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the exported model",
    )
    return parser.parse_args()


def main(args: Namespace) -> None:
    model_config = args.model
    logging.info(model_config)

    _, processor, model = load_model(model_config)
    model.eval()
    logging.info(f"{model_config.arch} model loaded from {model_config.pretrained}")

    # Determine input shape and input name
    input_shape = get_input_shape(model_config.arch, processor, args.batch_size)
    dummy_input = torch.randn(*input_shape)

    if model_config.arch in RGB_BASED_MODELS:
        input_names = ["pixel_values"]
    elif model_config.arch in POSE_BASED_MODELS:
        input_names = ["poses"]
    else:
        input_names = ["input"]

    output_names = ["logits"]

    # Output path
    output_path = args.output or f"{model_config.arch}.onnx"

    logging.info(f"Input shape: {input_shape}")
    logging.info(f"Exporting to {output_path}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            input_names[0]: {0: "batch_size"},
            output_names[0]: {0: "batch_size"},
        },
    )

    logging.info(f"Model exported to ONNX: {output_path}")

    # Verify the exported model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logging.info("ONNX model verified successfully")


if __name__ == "__main__":
    args = get_args()
    config_logger()
    main(args)
