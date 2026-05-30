import json
import evaluate
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import NamedTuple
from thop import profile
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


accuracy = evaluate.load("accuracy")


def compute_flops_and_params(model, inputs: dict) -> tuple:
    if "pose" in inputs:
        inputs = inputs["pose"]
    else:
        inputs = inputs["video"].permute(1, 0, 2, 3)
    inputs = inputs.unsqueeze(0).to(model.device)
    macs, params = profile(model, inputs=(inputs,), verbose=False)
    flops = macs * 2
    return flops, params


def save_evaluation_results(
    results: NamedTuple,
    classes: list,
    output_dir: Path,
    include_confusion_matrix: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = np.argmax(results.predictions, axis=1)
    references = results.label_ids
    results = {
        "metrics": results.metrics,
        "predictions": predictions.tolist(),
        "references": results.label_ids.tolist(),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    if include_confusion_matrix:
        compute_confusion_matrix(
            references,
            predictions,
            classes=classes,
            output_dir=output_dir,
            is_normalized=False,
        )
        compute_confusion_matrix(
            references,
            predictions,
            classes=classes,
            output_dir=output_dir,
            is_normalized=True,
        )


def compute_confusion_matrix(
    references: np.ndarray,
    predictions: np.ndarray,
    classes: list,
    output_dir: Path,
    is_normalized: bool = False,
) -> None:
    normalize = "all" if is_normalized else None
    cm = confusion_matrix(
        references,
        predictions,
        labels=range(len(classes)),
        normalize=normalize,
    )

    plt.figure(figsize=(1000, 1000))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        # display_labels=classes,
    )
    disp.plot(cmap=plt.cm.Blues)
    output_name = "normalized_confusion_matrix" if is_normalized else "confusion_matrix"
    plt.savefig(output_dir / f"{output_name}.png")
    plt.close()


def top_k_accuracy(eval_pred, k: int = 3) -> dict:
    top_k_preds = np.argsort(eval_pred.predictions, axis=1)[:, -k:]
    top_k_correct = np.any(top_k_preds == eval_pred.label_ids[:, None], axis=1)
    top_k_accuracy = np.mean(top_k_correct)
    return {f"top_{k}_accuracy": np.mean(top_k_accuracy)}


def compute_metrics(eval_pred):
    scores = {}

    predictions = np.argmax(eval_pred.predictions, axis=1)
    scores.update(
        accuracy.compute(
            predictions=predictions,
            references=eval_pred.label_ids,
        )
    )
    scores.update(
        {
            "f1": f1_score(
                y_true=eval_pred.label_ids,
                y_pred=predictions,
                average="macro",
                zero_division=0,
            )
        }
    )
    scores.update(
        {
            "recall": recall_score(
                y_true=eval_pred.label_ids,
                y_pred=predictions,
                average="macro",
                zero_division=0,
            )
        }
    )
    scores.update(
        {
            "precision": precision_score(
                y_true=eval_pred.label_ids,
                y_pred=predictions,
                average="macro",
                zero_division=0,
            )
        }
    )

    scores.update(top_k_accuracy(eval_pred, k=5))
    scores.update(top_k_accuracy(eval_pred, k=10))

    return scores
