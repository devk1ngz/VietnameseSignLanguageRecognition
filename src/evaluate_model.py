import shutil
import logging
from argparse import Namespace
from simple_parsing import ArgumentParser
from transformers import Trainer, TrainingArguments
from configs import DataConfig, EvaluationConfig
from utils import (
    compute_metrics,
    config_logger,
    TrainingCallback,
    save_evaluation_results,
    upload_to_hf,
)
from tools import (
    load_model,
    load_dataset,
    rgb_collate_fn,
    pose_collate_fn,
)


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Evaluate a SLR model",
        add_config_path_arg=True,
    )
    parser.add_arguments(DataConfig, "data")
    parser.add_arguments(EvaluationConfig, "evaluation")
    return parser.parse_args()


def main(args: Namespace) -> None:
    data_config = args.data
    logging.info(data_config)
    eval_config = args.evaluation
    logging.info(eval_config)

    dataset = load_dataset(data_config)
    logging.info(f"{data_config.dataset.upper()} dataset loaded")

    _, processor, model = load_model(eval_config)
    logging.info(f"{eval_config.arch} model loaded from {eval_config.pretrained}")

    eval_dataset = dataset.get_split(eval_config.eval_set, processor)
    logging.info("Splits created for training and evaluation")
    logging.info(f"Number of samples in {eval_config.eval_set} set: {eval_dataset.num_videos}")

    training_args = TrainingArguments(
        output_dir=eval_config.output_dir,
        remove_unused_columns=False,
        per_device_eval_batch_size=eval_config.batch_size,
    )
    data_collator = rgb_collate_fn if data_config.modality == "rgb" else pose_collate_fn
    callbacks = [TrainingCallback()]
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    logging.info("Evaluator created")

    # TODO: Add support for FLOPs calculation
    logging.info(f"FLOPs: {trainer.floating_point_ops(next(iter(eval_dataset)))}")

    num_trainable_params = trainer.get_num_trainable_parameters()
    logging.info(f"Number of trainable parameters: {num_trainable_params:,}")

    if eval_config.eval_set == "validation":
        metric_key_prefix = "val"
    else:
        metric_key_prefix = eval_config.eval_set

    logging.info("Evaluation started")
    results = trainer.predict(eval_dataset, metric_key_prefix=metric_key_prefix)
    logging.info(f"Results: {results.metrics}")
    save_evaluation_results(
        results=results,
        classes=dataset.gloss2id.keys(),
        output_dir=eval_config.output_dir,
    )
    logging.info(f"Results saved to {eval_config.output_dir}")
    if eval_config.push_to_hub:
        upload_to_hf(
            repo_id=eval_config.pretrained,
            path=eval_config.output_dir,
            path_in_repo=f"{eval_config.eval_set}/{data_config.dataset}",
            repo_type="model",
        )
        logging.info(f"Results uploaded to: {eval_config.pretrained}")
    logging.info("Evaluation completed")


if __name__ == "__main__":
    args = get_args()
    args.evaluation.output_dir = args.evaluation.output_dir / args.data.dataset
    args.evaluation.output_dir.mkdir(parents=True, exist_ok=True)

    config_logger(log_file=args.evaluation.output_dir / "evaluation.log")
    logging.info(f"Config file loaded from {args.config_path[0]}")

    shutil.copy(args.config_path[0], args.evaluation.output_dir / "evaluation.yaml")
    logging.info(f"Config file saved to {args.evaluation.output_dir}")

    main(args=args)
