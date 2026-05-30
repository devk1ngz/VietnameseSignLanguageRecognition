import os
import shutil
import logging
from argparse import Namespace
from simple_parsing import ArgumentParser
from transformers import TrainingArguments, Trainer
from configs import DataConfig, ModelConfig, TrainingConfig
from utils import (
    compute_metrics,
    config_logger,
    TrainingCallback,
    save_evaluation_results,
    upload_to_hf,
    compute_flops_and_params,
)
from tools import (
    load_model,
    load_dataset,
    rgb_collate_fn,
    pose_collate_fn
)


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Train a SLR model",
        add_config_path_arg=True,
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="VieSignLang",
        help="Weights and Biases entity",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="sign-language-recognition",
        help="Weights and Biases project",
    )
    parser.add_arguments(DataConfig, "data")
    parser.add_arguments(ModelConfig, "model")
    parser.add_arguments(TrainingConfig, "training")
    return parser.parse_args()


def main(args: Namespace) -> None:
    data_config = args.data
    logging.info(data_config)
    model_config = args.model
    logging.info(model_config)
    training_config = args.training
    logging.info(training_config)

    dataset = load_dataset(data_config)
    logging.info(f"{data_config.dataset.upper()} dataset loaded")

    config, processor, model = load_model(
        model_config,
        label2id=dataset.gloss2id,
        id2label=dataset.id2gloss,
        do_train=True,
    )
    logging.info(f"{model_config.arch} model loaded from {model_config.pretrained}")

    train_dataset = dataset.get_split("train", processor)
    val_dataset = dataset.get_split("validation", processor)
    test_dataset = dataset.get_split("test", processor)
    logging.info("Splits created for training and evaluation")
    logging.info(f"Number of samples in training set: {train_dataset.num_videos}")
    logging.info(f"Number of samples in validation set: {val_dataset.num_videos}")
    logging.info(f"Number of samples in test set: {test_dataset.num_videos}")

    if data_config.modality == "rgb":
        training_samples = train_dataset.num_videos
        batch_size = training_config.per_device_train_batch_size
        num_epochs = training_config.num_train_epochs
        training_config.max_steps = (training_samples // batch_size) * num_epochs
        data_collator = rgb_collate_fn
    else:
        data_collator = pose_collate_fn

    callbacks = [TrainingCallback()]
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**vars(training_config)),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=callbacks,
        tokenizer=processor,
    )
    logging.info("Trainer created")

    flops, params = compute_flops_and_params(model, next(iter(train_dataset)))
    logging.info(f"FLOPs: {flops:,}")
    logging.info(f"Number of parameters: {params:,}")

    logging.info("Training started")
    trainer.train()
    logging.info("Training completed")

    trainer.save_model(training_config.output_dir)
    logging.info(f"Model saved to {training_config.output_dir}")

    logging.info("Evaluation started")

    val_output_dir = training_config.output_dir / "validation" / data_config.dataset
    val_results = trainer.predict(test_dataset, metric_key_prefix="val")
    logging.info(f"Validation results: {val_results.metrics}")
    save_evaluation_results(
        results=val_results,
        classes=dataset.gloss2id.keys(),
        output_dir=val_output_dir,
    )
    logging.info(f"Validation results saved to {val_output_dir}")

    test_output_dir = training_config.output_dir / "test" / data_config.dataset
    test_results = trainer.predict(test_dataset, metric_key_prefix="test")
    logging.info(f"Test results: {test_results.metrics}")
    save_evaluation_results(
        results=test_results,
        classes=dataset.gloss2id.keys(),
        output_dir=test_output_dir,
    )
    logging.info(f"Test results saved to {val_output_dir}")

    if training_config.push_to_hub:
        upload_to_hf(
            repo_id=training_config.hub_model_id,
            path=val_output_dir,
            path_in_repo=f"validation/{data_config.dataset}",
            repo_type="model",
        )
        upload_to_hf(
            repo_id=training_config.hub_model_id,
            path=test_output_dir,
            path_in_repo=f"test/{data_config.dataset}",
            repo_type="model",
        )
        logging.info(f"Results pushed to {training_config.hub_model_id}")
    logging.info("Evaluation completed")

    if training_config.hub_model_id:
        config.label2id = dataset.gloss2id
        config.id2label = dataset.id2gloss
        config.push_to_hub(training_config.hub_model_id)
        logging.info(f"Model config pushed to {training_config.hub_model_id}")


if __name__ == "__main__":
    args = get_args()
    config_logger(log_file=args.training.output_dir / "train.log")
    logging.info(f"Config file loaded from {args.config_path[0]}")

    shutil.copy(args.config_path[0], args.training.output_dir / "train.yaml")
    logging.info(f"Config file saved to {args.training.output_dir}")

    if args.training.report_to == "wandb":
        os.environ["WANDB_MODE"] = "run"
        os.environ["WANDB_ENTITY"] = args.wandb_entity
        os.environ["WANDB_PROJECT"] = args.wandb_project
        logging.info(f"Wandb entity: {args.wandb_entity}")
        logging.info(f"Wandb project: {args.wandb_project}")
    else:
        os.environ["WANDB_MODE"] = "disabled"

    main(args=args)
