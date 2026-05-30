import logging
from tools import register_pipeline
from configs import ModelConfig
from simple_parsing import ArgumentParser
from argparse import Namespace
from utils import config_logger


def get_args() -> Namespace:
    parser = ArgumentParser(description="Train a model on VSL")
    parser.add_arguments(ModelConfig, "model")
    return parser.parse_args()


def main(args: Namespace) -> None:
    model_config = args.model
    logging.info(model_config)

    pipeline = register_pipeline(model_config)
    logging.info("Pipeline loaded")

    pipeline.push_to_hub(model_config.pretrained)
    logging.info("Pipeline pushed to HuggingFace")


if __name__ == "__main__":
    args = get_args()
    config_logger()
    main(args)
