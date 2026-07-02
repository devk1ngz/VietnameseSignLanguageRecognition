"""
Export the SL-GCN multi-stream ensemble as a SINGLE ONNX model.

The multi-stream ensemble is natively 4 separate models + a softmax average.
This script fuses them into one graph so deployment only ever loads/runs one
ONNX file:

    raw joint tensor  ->  [ joint / bone / joint-motion / bone-motion transforms
                            computed INSIDE the graph ]  ->  4 SL-GCN sub-models
                         ->  average softmax  ->  probabilities

Input contract
--------------
The ONNX input ``poses`` is the joint tensor **after joint-selection + padding
but BEFORE normalization / bone / motion** — i.e. exactly the output of
``SLGCNPad`` (shape ``(B, 3, T, 27, 1)``). All stream-specific transforms and the
normalization are done inside the graph, so deployment preprocessing is just
``PoseExtract -> SLGCNJointSelect -> SLGCNPad``.

Output
------
``probabilities`` of shape ``(B, num_classes)`` — the ensembled class
probabilities. Take ``argmax`` for the predicted gloss id.

Run from ``src/`` (needs HF_HOME so the model's remote code can import):

    HF_HOME=../hf_cache PATH=../.venv/bin:$PATH \
    python convert_ensemble_to_onnx.py \
        --streams \
            experiments/sl_gcn_joint_multicam \
            experiments/sl_gcn_bone_multicam \
            experiments/sl_gcn_joint_motion_multicam \
            experiments/sl_gcn_bone_motion_multicam \
        --output sl_gcn_ensemble.onnx
"""
import argparse
import logging

import torch
import torch.nn as nn

from tools.models import load_pose_pretrained, read_gloss2id
from models.sl_gcn.modelling import _build_edges


def build_parent_index(num_points: int) -> list:
    """Parent joint index for each joint (root maps to itself -> zero bone).

    Mirrors ``features/transforms/sl_gcn.py:_build_bone_pairs`` (BFS tree rooted
    at joint 0) but returns a full-length parent map so the bone stream can be a
    single ``index_select`` + subtract (ONNX-friendly, no scatter).
    """
    edges = _build_edges(num_points)
    neighbours = {v: [] for v in range(num_points)}
    for i, j in edges:
        neighbours[i].append(j)
        neighbours[j].append(i)
    parent = {0: None}
    queue = [0]
    while queue:
        node = queue.pop(0)
        for nxt in neighbours[node]:
            if nxt not in parent:
                parent[nxt] = node
                queue.append(nxt)
    parent_full = list(range(num_points))  # default: self (root -> zero bone)
    for child, par in parent.items():
        if par is not None:
            parent_full[child] = par
    return parent_full


class SLGCNEnsemble(nn.Module):
    """Fuses N SL-GCN stream models + their input transforms into one module.

    Input  ``poses``: (B, C=3, T, V=num_points, M=num_people), post-pad / pre-norm.
    Output ``probs``: (B, num_classes), averaged softmax over the streams.
    """

    def __init__(self, stream_models, stream_specs, num_points: int):
        super().__init__()
        self.models = nn.ModuleList(stream_models)
        # bools stored as plain python (baked into the trace)
        self.stream_specs = stream_specs
        self.register_buffer(
            "parent_full", torch.tensor(build_parent_index(num_points), dtype=torch.long)
        )

    def _bone(self, x):  # x: (B,C,T,V,M); V is dim 3
        # bone[j] = x[j] - x[parent[j]]; root(0)->0 (parent is itself)
        return x - x.index_select(3, self.parent_full)

    def _motion(self, x):  # displacement along time (dim 2); last frame stays 0
        motion = torch.zeros_like(x)
        motion[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
        return motion

    def _normalize(self, x):
        # Matches SLGCNNormalize(is_vector=False): subtract, per sample, the
        # mean-over-time of channel 0/1 at joint 0; channel 2 (confidence) kept.
        m0 = x[:, 0:1, :, 0:1, 0:1].mean(dim=2, keepdim=True)  # (B,1,1,1,1)
        m1 = x[:, 1:2, :, 0:1, 0:1].mean(dim=2, keepdim=True)
        zero = torch.zeros_like(m0)
        offset = torch.cat([m0, m1, zero], dim=1)              # (B,3,1,1,1)
        return x - offset

    def forward(self, poses):
        prob_sum = None
        for model, (bone, motion) in zip(self.models, self.stream_specs):
            x = poses
            if bone:
                x = self._bone(x)
            if motion:
                x = self._motion(x)
            x = self._normalize(x)
            logits = model(x)                    # inner SLGCN returns a tensor
            probs = torch.softmax(logits, dim=1)
            prob_sum = probs if prob_sum is None else prob_sum + probs
        return prob_sum / len(self.models)


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--streams", nargs="+", required=True,
                   help="Trained per-stream model dirs (order does not matter)")
    p.add_argument("--gloss-csv", default="../data/processed/vsl_400/gloss.csv",
                   help="id,gloss csv to backfill the checkpoints' null label2id")
    p.add_argument("--output", default="sl_gcn_ensemble.onnx")
    p.add_argument("--opset-version", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--no-check", action="store_true",
                   help="Skip the onnxruntime numerical parity check")
    return p.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    gloss2id = read_gloss2id(args.gloss_csv)
    logging.info(f"gloss.csv: {len(gloss2id)} classes")

    models, specs = [], []
    ref_cfg = None
    for path in args.streams:
        cfg, _, wrapper = load_pose_pretrained(path, label2id=gloss2id)
        inner = wrapper.model                    # the bare SLGCN (returns logits)
        models.append(inner)
        specs.append((bool(cfg.bone_stream), bool(cfg.motion_stream)))
        logging.info(
            f"loaded {path}  bone={cfg.bone_stream} motion={cfg.motion_stream} "
            f"classes={len(cfg.id2label)}"
        )
        if ref_cfg is None:
            ref_cfg = cfg
        else:
            assert cfg.num_points == ref_cfg.num_points, "num_points mismatch"
            assert len(cfg.id2label) == len(ref_cfg.id2label), "num_classes mismatch"
            assert cfg.window_size == ref_cfg.window_size, "window_size mismatch"

    ensemble = SLGCNEnsemble(models, specs, ref_cfg.num_points).eval()

    input_shape = (
        args.batch_size,
        ref_cfg.in_channels,
        ref_cfg.window_size,
        ref_cfg.num_points,
        ref_cfg.num_people,
    )
    dummy = torch.randn(*input_shape)
    logging.info(f"input shape (poses): {input_shape}")

    with torch.no_grad():
        torch_out = ensemble(dummy)
    logging.info(f"output shape (probabilities): {tuple(torch_out.shape)}")

    torch.onnx.export(
        ensemble,
        dummy,
        args.output,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=["poses"],
        output_names=["probabilities"],
        dynamic_axes={"poses": {0: "batch_size"},
                      "probabilities": {0: "batch_size"}},
        dynamo=False,  # use the stable TorchScript exporter (no onnxscript dep)
    )
    logging.info(f"Exported ensemble ONNX -> {args.output}")

    import onnx
    onnx.checker.check_model(onnx.load(args.output))
    logging.info("ONNX structure verified")

    if not args.no_check:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        ort_out = sess.run(["probabilities"], {"poses": dummy.numpy()})[0]
        max_diff = float(np.abs(ort_out - torch_out.numpy()).max())
        logging.info(f"torch vs onnxruntime max abs diff: {max_diff:.2e}")
        assert max_diff < 1e-4, "ONNX output diverges from PyTorch!"
        logging.info("Numerical parity OK — single ONNX ensemble is ready.")


if __name__ == "__main__":
    main()
