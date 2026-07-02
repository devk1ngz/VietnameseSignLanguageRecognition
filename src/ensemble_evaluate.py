"""
Multi-stream SL-GCN ensemble evaluation.

Loads several separately-trained stream models (joint / bone / joint-motion /
bone-motion), runs each over the SAME split, averages their softmax
probabilities (late fusion), and reports the ensembled metrics — this is how
the original SL-GCN paper reaches its best accuracy.

Each stream applies its own input transform automatically: the bone/motion
flags are read from every model's ``preprocessor_config.json``, so all we do
here is load the model + its processor and let ``get_split`` build the right
pipeline.

Run from ``src/``:

    HF_HOME=../hf_cache PATH=../.venv/bin:$PATH CUDA_VISIBLE_DEVICES=1 \
    python ensemble_evaluate.py \
        --data_dir ../data/processed/vsl_400 \
        --subset cam_1_2_3 \
        --eval_set test \
        --model_paths \
            experiments/sl_gcn_joint_multicam \
            experiments/sl_gcn_bone_multicam \
            experiments/sl_gcn_joint_motion_multicam \
            experiments/sl_gcn_bone_motion_multicam
"""
import json
import logging
import argparse
from pathlib import Path
from collections import namedtuple

import numpy as np
from transformers import Trainer, TrainingArguments

from configs import DataConfig
from tools import load_dataset, pose_collate_fn
from tools.models import load_pose_pretrained
from utils import compute_metrics, config_logger, save_evaluation_results

# Mimic the HF EvalPrediction interface expected by compute_metrics /
# save_evaluation_results (they only touch .predictions / .label_ids / .metrics).
EnsemblePrediction = namedtuple(
    "EnsemblePrediction", ["predictions", "label_ids", "metrics"]
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SL-GCN multi-stream ensemble eval")
    parser.add_argument("--model_paths", nargs="+", required=True,
                        help="Paths to the trained per-stream model dirs")
    parser.add_argument("--dataset", default="visl_400", choices=["visl_98", "visl_400"])
    parser.add_argument("--data_dir", default="../data/processed/vsl_400")
    parser.add_argument("--subset", default="cam_1_2_3")
    parser.add_argument("--eval_set", default="test",
                        choices=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", default="experiments/sl_gcn_ensemble_multicam")
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predict_stream(model_path: str, dataset, eval_set: str,
                   batch_size: int, output_dir: Path):
    """Return (probs, label_ids) for a single stream model."""
    # Checkpoints save null label2id; backfill from the dataset's gloss mapping.
    _, processor, model = load_pose_pretrained(
        model_path, label2id=dataset.gloss2id,
    )
    logging.info(
        f"[{model_path}] bone_stream={getattr(processor, 'bone_stream', None)} "
        f"motion_stream={getattr(processor, 'motion_stream', None)}"
    )

    # The processor's bone/motion flags drive the transform pipeline, so each
    # stream sees the representation it was trained on.
    eval_dataset = dataset.get_split(eval_set, processor)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            remove_unused_columns=False,
            per_device_eval_batch_size=batch_size,
            report_to=[],
        ),
        data_collator=pose_collate_fn,
        compute_metrics=compute_metrics,
    )
    prefix = "val" if eval_set == "validation" else eval_set
    out = trainer.predict(eval_dataset, metric_key_prefix=prefix)
    logging.info(f"[{model_path}] single-stream metrics: {out.metrics}")
    return softmax(out.predictions), out.label_ids


def main() -> None:
    args = get_args()
    output_dir = Path(args.output_dir) / args.eval_set / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    config_logger(log_file=output_dir / "ensemble.log")
    logging.info(f"Ensembling {len(args.model_paths)} streams: {args.model_paths}")

    data_config = DataConfig(
        dataset=args.dataset,
        modality="pose",
        subset=args.subset,
        data_dir=args.data_dir,
        debug=False,
    )
    dataset = load_dataset(data_config)
    logging.info(f"{args.dataset.upper()} dataset loaded")

    prob_sum = None
    ref_labels = None
    per_stream = {}
    for model_path in args.model_paths:
        probs, labels = predict_stream(
            model_path, dataset, args.eval_set, args.batch_size, output_dir,
        )
        if ref_labels is None:
            ref_labels = labels
        else:
            # All streams iterate the same split in the same (sequential) order,
            # so label alignment must hold — guard against a silent mismatch.
            assert np.array_equal(ref_labels, labels), \
                f"Label order mismatch for {model_path}; cannot ensemble."
        prob_sum = probs if prob_sum is None else prob_sum + probs
        # Track this stream's own accuracy for the summary table.
        acc = float((probs.argmax(axis=1) == labels).mean())
        per_stream[Path(model_path).name] = acc

    avg_probs = prob_sum / len(args.model_paths)
    ensemble_metrics = compute_metrics(
        EnsemblePrediction(predictions=avg_probs, label_ids=ref_labels, metrics=None)
    )
    ensemble_metrics = {f"ensemble_{k}": v for k, v in ensemble_metrics.items()}

    logging.info("=== Per-stream accuracy ===")
    for name, acc in per_stream.items():
        logging.info(f"  {name}: {acc:.4f}")
    logging.info(f"=== ENSEMBLE metrics: {ensemble_metrics}")

    save_evaluation_results(
        results=EnsemblePrediction(
            predictions=avg_probs,
            label_ids=ref_labels,
            metrics=ensemble_metrics,
        ),
        classes=dataset.gloss2id.keys(),
        output_dir=output_dir,
    )
    with open(output_dir / "ensemble_summary.json", "w") as f:
        json.dump(
            {"per_stream_accuracy": per_stream, "ensemble_metrics": ensemble_metrics,
             "streams": args.model_paths},
            f, indent=4,
        )
    logging.info(f"Ensemble results saved to {output_dir}")


if __name__ == "__main__":
    main()
