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
        description="Convert a trained model to TorchScript format",
    )
    parser.add_arguments(ModelConfig, "model")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for TorchScript model. Defaults to '<arch>.pt'",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for tracing",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="trace",
        choices=["trace", "script"],
        help="TorchScript conversion method: 'trace' or 'script'",
    )
    return parser.parse_args()


def main(args: Namespace) -> None:
    model_config = args.model
    logging.info(model_config)

    _, processor, model = load_model(model_config)
    model.eval()
    logging.info(f"{model_config.arch} model loaded from {model_config.pretrained}")

    # Output path
    output_path = args.output or f"{model_config.arch}.pt"

    if args.method == "trace":
        # Determine input shape
        input_shape = get_input_shape(model_config.arch, processor, args.batch_size)
        dummy_input = torch.randn(*input_shape)

        logging.info(f"Input shape: {input_shape}")
        logging.info(f"Tracing model to {output_path}")

        traced_model = torch.jit.trace(model, dummy_input)
    else:
        logging.info(f"Scripting model to {output_path}")
        traced_model = torch.jit.script(model)

    traced_model.save(output_path)
    logging.info(f"TorchScript model saved to {output_path}")

    # Verify the exported model
    loaded_model = torch.jit.load(output_path)
    logging.info("TorchScript model loaded and verified successfully")

    if args.method == "trace":
        # Run a forward pass to verify
        with torch.no_grad():
            output = loaded_model(dummy_input)
        logging.info(f"Verification output shape: {output.logits.shape}")


if __name__ == "__main__":
    args = get_args()
    config_logger()
    main(args)
