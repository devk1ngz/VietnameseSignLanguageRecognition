#!/usr/bin/env python3
"""
Check training progress from an experiment's train.log + trainer_state.json.

Shows: progress (epoch/steps + %), ETA, latest train loss, latest & best eval
metrics (accuracy / top-5 / top-10 / loss), a recent-epoch trend table, whether
the run is still alive, and a comparison against the baseline test scores.

Usage (from anywhere):
    python scripts/check_training.py                       # auto-pick newest run
    python scripts/check_training.py spoter_v3.0_multicam  # by run name
    python scripts/check_training.py src/experiments/foo   # by path
    python scripts/check_training.py --watch 30            # refresh every 30s
    python scripts/check_training.py --last 15             # show 15 recent epochs

No third-party deps — stdlib only.
"""
import argparse
import ast
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Baseline to beat: sl_gcn_v2.0 single-joint run (measured).
#   VAL  best acc 0.8684   |   TEST acc 0.7653 / top5 0.9139
# We compare the current VAL best against the previous VAL best, and separately
# note the TEST target (spoter_v3.0 reaches 0.857 test — the number to chase).
BASELINE_VAL_TOP1 = 0.8684
BASELINE_TEST_TOP1 = 0.7653
BASELINE_TEST_TOP5 = 0.9139
SPOTER_TEST_TOP1 = 0.857

# The 4-stream ensemble pipeline (in training order).
ENSEMBLE_STREAMS = [
    "sl_gcn_joint_multicam",
    "sl_gcn_bone_multicam",
    "sl_gcn_joint_motion_multicam",
    "sl_gcn_bone_motion_multicam",
]
DEFAULT_SESSION = "sl_gcn_ens"

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "src" / "experiments"

LINE_RE = re.compile(r"\[(?P<ts>[\d/]+ [\d:]+)\].*(?P<dict>\{[^{}]*\})\s*$")
TS_FMT = "%m/%d/%Y %H:%M:%S"


def resolve_run(arg):
    """Return the experiment directory for a run name / path / auto-pick."""
    if arg:
        p = Path(arg)
        if p.is_dir():
            return p.resolve()
        cand = EXPERIMENTS_DIR / arg
        if cand.is_dir():
            return cand
        sys.exit(f"Run not found: {arg} (looked in {EXPERIMENTS_DIR})")
    # auto-pick: newest dir under experiments that has a train.log
    runs = [d for d in EXPERIMENTS_DIR.glob("*") if (d / "train.log").exists()]
    if not runs:
        sys.exit(f"No runs with train.log under {EXPERIMENTS_DIR}")
    return max(runs, key=lambda d: (d / "train.log").stat().st_mtime)


def parse_log(log_path):
    """Parse timestamped metric dicts. Returns (train_rows, eval_rows).

    Each row: (timestamp:datetime|None, metrics:dict).
    """
    train_rows, eval_rows = [], []
    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            if "Problematic normalization" in line:
                continue
            m = LINE_RE.search(line)
            if not m:
                continue
            try:
                d = ast.literal_eval(m.group("dict"))
            except (ValueError, SyntaxError):
                continue
            if not isinstance(d, dict):
                continue
            try:
                ts = dt.datetime.strptime(m.group("ts"), TS_FMT)
            except ValueError:
                ts = None
            if "eval_accuracy" in d:
                eval_rows.append((ts, d))
            elif "loss" in d:
                train_rows.append((ts, d))
    return train_rows, eval_rows


def latest_trainer_state(run_dir):
    """Load trainer_state.json from the newest checkpoint, if any."""
    ckpts = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda d: int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else -1,
    )
    for ck in reversed(ckpts):
        ts = ck / "trainer_state.json"
        if ts.exists():
            try:
                return json.load(open(ts))
            except json.JSONDecodeError:
                continue
    return None


def read_total_epochs(run_dir, state):
    if state and state.get("num_train_epochs"):
        return int(state["num_train_epochs"])
    # fallback: train.yaml
    y = run_dir / "train.yaml"
    if y.exists():
        m = re.search(r"num_train_epochs:\s*(\d+)", y.read_text())
        if m:
            return int(m.group(1))
    return None


def is_running(run_name):
    """True if a train.py process referencing this run/config is alive."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "args"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:
        return None
    for ln in out.splitlines():
        if "train.py" in ln and "check_training" not in ln:
            # match by run name or by config file stem if present
            return True
    return False


def gpu_status():
    """Return a list of per-GPU status strings via nvidia-smi (or None)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except Exception:
        return None
    rows = []
    for ln in out.strip().splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != 4:
            continue
        idx, util, used, total = parts
        try:
            used_gb = float(used) / 1024
            total_gb = float(total) / 1024
        except ValueError:
            continue
        busy = "●" if float(util or 0) > 5 else "○"
        rows.append(
            f"  {busy} GPU{idx}: {float(util or 0):>3.0f}% util   "
            f"{used_gb:5.1f} / {total_gb:4.0f} GB"
        )
    return rows


def tmux_live_step(session):
    """Read the current tqdm training bar from a tmux pane, without attaching.

    Returns (step, total, rate_str, remaining_str) or None. Picks the bar with
    the largest total (the training bar, not the per-epoch eval bar).
    """
    try:
        has = subprocess.run(
            ["tmux", "has-session", "-t", session],
            capture_output=True, timeout=5,
        ).returncode
        if has != 0:
            return None
        pane = subprocess.run(
            ["tmux", "capture-pane", "-t", session, "-p", "-S", "-2000"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return None
    pane = pane.replace("\r", "\n")
    bars = re.findall(r"(\d+)/(\d+) \[([^\]]*)\]", pane)
    if not bars:
        return None
    max_total = max(int(t) for _, t, _ in bars)
    # last-printed bar with that (largest) total = most recent training step
    step = total = None
    inside = ""
    for s, t, ins in bars:
        if int(t) == max_total:
            step, total, inside = int(s), int(t), ins
    remaining = inside.split("<")[-1].split(",")[0] if "<" in inside else "—"
    rate = inside.split(", ")[-1] if ", " in inside else "—"
    return step, total, rate, remaining


def ensemble_overview(current_run):
    """Compact status of all 4 streams in the ensemble pipeline."""
    lines = []
    any_present = any((EXPERIMENTS_DIR / s / "train.log").exists()
                      for s in ENSEMBLE_STREAMS)
    if not any_present:
        return lines
    lines.append("ENSEMBLE PIPELINE (4 streams)")
    for i, name in enumerate(ENSEMBLE_STREAMS, 1):
        run_dir = EXPERIMENTS_DIR / name
        short = name.replace("sl_gcn_", "").replace("_multicam", "")
        if not (run_dir / "train.log").exists():
            lines.append(f"  {i}. {short:<12} ⋯ pending")
            continue
        best = best_val_of(name)
        _, eval_rows = parse_log(run_dir / "train.log")
        last_ep = eval_rows[-1][1].get("epoch") if eval_rows else 0
        done = (run_dir / "model.safetensors").exists() and name != current_run
        mark = "▶ running" if name == current_run else ("✓ done" if done else "· started")
        best_str = f"best val {fmt_pct(best)}" if best is not None else "no epoch yet"
        lines.append(f"  {i}. {short:<12} {mark:<10} ep{last_ep:>4g}  {best_str}")
    lines.append("")
    return lines


def best_val_of(run_name):
    """Best eval accuracy of another run (for v1 vs v2 comparison), or None."""
    run_dir = EXPERIMENTS_DIR / run_name
    if not run_dir.is_dir():
        return None
    state = latest_trainer_state(run_dir)
    if state and state.get("best_metric") is not None:
        return state["best_metric"]
    _, eval_rows = parse_log(run_dir / "train.log")
    if eval_rows:
        return max(d["eval_accuracy"] for _, d in eval_rows)
    return None


def fmt_pct(x):
    return f"{x * 100:.2f}%"


def fmt_eta(seconds):
    if seconds is None or seconds <= 0:
        return "—"
    td = dt.timedelta(seconds=int(seconds))
    return str(td)


def avg_epoch_seconds(train_rows):
    """Mean wall-clock seconds between consecutive train (per-epoch) entries."""
    deltas = []
    for (t0, _), (t1, _) in zip(train_rows, train_rows[1:]):
        if t0 and t1:
            d = (t1 - t0).total_seconds()
            if 0 < d < 24 * 3600:
                deltas.append(d)
    if not deltas:
        return None
    # use last few epochs for a more current estimate
    recent = deltas[-5:]
    return sum(recent) / len(recent)


def render(run_dir, last_n, session=DEFAULT_SESSION):
    log_path = run_dir / "train.log"
    train_rows, eval_rows = parse_log(log_path)
    state = latest_trainer_state(run_dir)
    total_epochs = read_total_epochs(run_dir, state)
    live = tmux_live_step(session)

    lines = []
    lines.append("=" * 64)
    lines.append(f"  RUN: {run_dir.name}")
    lines.append(f"  log: {log_path}")
    running = is_running(run_dir.name)
    status = "🟢 RUNNING" if running else "⚪ no train.py process found"
    lines.append(f"  status: {status}   (checked {dt.datetime.now():%H:%M:%S})")
    lines.append("=" * 64)

    # ----- GPU panel -----
    gpus = gpu_status()
    if gpus:
        lines.append("GPUs")
        lines.extend(gpus)
        lines.append("")

    # ----- ensemble pipeline overview -----
    overview = ensemble_overview(run_dir.name)
    if overview:
        lines.extend(overview)

    if not train_rows and not eval_rows and not live:
        lines.append("No metric lines yet — training may still be warming up.")
        return "\n".join(lines)

    # ----- progress -----
    # Completed epoch (from log, updates only at epoch end).
    done_epoch = None
    if train_rows:
        done_epoch = train_rows[-1][1].get("epoch")
    elif eval_rows:
        done_epoch = eval_rows[-1][1].get("epoch")

    # Live step/epoch from the tmux pane (updates continuously, even mid-epoch).
    g_step = live[0] if live else (state.get("global_step") if state else None)
    max_step = live[1] if live else (state.get("max_steps") if state else None)
    live_epoch = None
    if g_step and max_step and total_epochs:
        live_epoch = g_step / (max_step / total_epochs)

    lines.append("PROGRESS")
    # Prefer the live epoch (real-time); fall back to the last completed epoch.
    cur_epoch = live_epoch if live_epoch is not None else done_epoch
    if cur_epoch is not None and total_epochs:
        pct = cur_epoch / total_epochs
        suffix = " (live)" if live_epoch is not None else ""
        lines.append(
            f"  epoch   : {cur_epoch:.2f} / {total_epochs}   ({fmt_pct(pct)}){suffix}"
        )
    elif cur_epoch is not None:
        lines.append(f"  epoch   : {cur_epoch:g}")
    if g_step and max_step:
        src = "live" if live else "checkpoint"
        lines.append(
            f"  steps   : {g_step} / {max_step}   ({fmt_pct(g_step / max_step)}) [{src}]"
        )
    if live:
        lines.append(f"  rate    : {live[2]}   time left (this stream): {live[3]}")

    eps = avg_epoch_seconds(train_rows)
    if eps and cur_epoch is not None and total_epochs:
        remaining = (total_epochs - cur_epoch) * eps
        eta_done = dt.datetime.now() + dt.timedelta(seconds=remaining)
        lines.append(
            f"  speed   : ~{eps:.1f}s/epoch   ETA: {fmt_eta(remaining)}  "
            f"(done ~{eta_done:%a %H:%M})"
        )

    # ----- latest train loss -----
    if train_rows:
        d = train_rows[-1][1]
        lines.append("")
        lines.append("LATEST TRAIN")
        lines.append(
            f"  loss={d.get('loss'):.4f}  lr={d.get('learning_rate'):.2e}  "
            f"grad_norm={d.get('grad_norm', float('nan')):.3f}  epoch={d.get('epoch'):g}"
        )

    # ----- latest eval -----
    if eval_rows:
        d = eval_rows[-1][1]
        lines.append("")
        lines.append("LATEST EVAL (val split)")
        lines.append(
            f"  acc={fmt_pct(d['eval_accuracy'])}  "
            f"top5={fmt_pct(d.get('eval_top_5_accuracy', 0))}  "
            f"top10={fmt_pct(d.get('eval_top_10_accuracy', 0))}"
        )
        lines.append(
            f"  loss={d.get('eval_loss', float('nan')):.4f}  "
            f"f1={d.get('eval_f1', 0):.4f}  epoch={d.get('epoch'):g}"
        )

    # ----- best so far -----
    best_acc, best_epoch, best_top5 = -1, None, None
    for _, d in eval_rows:
        if d["eval_accuracy"] > best_acc:
            best_acc = d["eval_accuracy"]
            best_epoch = d.get("epoch")
            best_top5 = d.get("eval_top_5_accuracy")
    if best_acc >= 0:
        lines.append("")
        lines.append("BEST EVAL SO FAR")
        bc = state.get("best_model_checkpoint") if state else None
        lines.append(
            f"  acc={fmt_pct(best_acc)}  top5={fmt_pct(best_top5 or 0)}  "
            f"@ epoch {best_epoch:g}" + (f"   [{Path(bc).name}]" if bc else "")
        )
        # apples-to-apples: this is VAL, compare against sl_gcn_v2.0 VAL best.
        arrow = lambda x: "▲" if x >= 0 else "▼"
        dv2 = best_acc - BASELINE_VAL_TOP1
        lines.append(
            f"  vs sl_gcn_v2.0 VAL best {fmt_pct(BASELINE_VAL_TOP1)}:  "
            f"{arrow(dv2)}{fmt_pct(abs(dv2))}  (this stream, VAL split)"
        )
        lines.append(
            f"  TEST targets — beat sl_gcn_v2.0 {fmt_pct(BASELINE_TEST_TOP1)}, "
            f"chase spoter_v3.0 {fmt_pct(SPOTER_TEST_TOP1)} "
            f"(see ensemble test results when pipeline finishes)"
        )

    # ----- recent trend -----
    if eval_rows:
        lines.append("")
        lines.append(f"RECENT EPOCHS (last {min(last_n, len(eval_rows))})")
        lines.append(f"  {'epoch':>6}  {'acc':>8}  {'top5':>8}  {'top10':>8}  {'eval_loss':>9}")
        for _, d in eval_rows[-last_n:]:
            lines.append(
                f"  {d.get('epoch', 0):>6g}  {fmt_pct(d['eval_accuracy']):>8}  "
                f"{fmt_pct(d.get('eval_top_5_accuracy', 0)):>8}  "
                f"{fmt_pct(d.get('eval_top_10_accuracy', 0)):>8}  "
                f"{d.get('eval_loss', float('nan')):>9.4f}"
            )

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="run name or experiment dir (default: newest)")
    ap.add_argument("--last", type=int, default=10, help="recent eval epochs to show")
    ap.add_argument("--watch", type=int, metavar="SEC", help="refresh every SEC seconds")
    ap.add_argument("--session", default=DEFAULT_SESSION,
                    help=f"tmux session for live step (default: {DEFAULT_SESSION})")
    args = ap.parse_args()

    if args.watch:
        try:
            while True:
                # re-resolve each tick so it follows the pipeline to the next stream
                run_dir = resolve_run(args.run)
                os.system("clear")
                print(render(run_dir, args.last, args.session))
                print(f"\n(watching every {args.watch}s — Ctrl+C to stop)")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nstopped.")
    else:
        run_dir = resolve_run(args.run)
        print(render(run_dir, args.last, args.session))


if __name__ == "__main__":
    main()
