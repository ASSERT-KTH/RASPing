"""Initialization-control comparison: buggy-init repair vs. random-init repair.

Addresses the reviewer request to compare buggy-program initialization
against random initialization of the same architecture, with learning
curves and sample efficiency. (The "unrelated compiled program"
initialization control from the same comment was checked and ruled out:
RASP-to-transformer compilation produces a different architecture per
program, so weights aren't transplantable between them - see
launch_random_init_baseline.py's docstring.)

Two data sources, both produced by the *same* train_mutations.py /
test_trained_mutations.py pipeline (so file formats match exactly):
  - buggy-init: saved_data/{program}/cross_entropy_loss/job_{job_id}/
    (existing mutant-repair results; a stratified-by-mutation-order subset
    is selected here, not all of them, to keep the plot readable)
  - random-init: saved_data_random_init/{program}/seed_{seed}/cross_entropy_loss/job_seed{seed}/
    (produced by launch_random_init_baseline.py)

Usage:
    python plot_initialization_controls.py --subset-size 15
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).parent
BUGGY_SAVED_DATA = HERE / "saved_data"
RANDOM_INIT_SAVED_DATA = HERE / "saved_data_random_init"
MUTATIONS_PATH = HERE.parent / "mutation" / "results" / "aggregated_mutations.json"
PLOTS_DIR = HERE.parent.parent / "plots"

PROGRAMS = ["hist", "most_freq", "reverse", "sort"]
LOSS_FN = "cross_entropy_loss"
VAL_STEP = 10  # must match train_mutations.py's Trainer(valStep=...)
N_SEEDS = 10


def select_buggy_init_subset(program: str, subset_size: int, seed: int = 0) -> list[str]:
    """Stratified sample of job_ids by mutation_order, restricted to jobs with saved results."""
    df = pd.read_json(MUTATIONS_PATH)
    buggy = df[df["execution_result"].apply(lambda r: r.get("status") == "BUGGY_MODEL")]
    buggy = buggy[buggy["program_name"] == program]

    saved_dir = BUGGY_SAVED_DATA / program / LOSS_FN
    completed = set()
    if saved_dir.exists():
        for d in saved_dir.iterdir():
            if (d / "train_losses.npy").exists() and (d / "test_results.json").exists():
                completed.add(d.name.replace("job_", ""))

    buggy = buggy[buggy["job_id"].isin(completed)]
    if buggy.empty:
        return []

    rng = np.random.default_rng(seed)
    orders = sorted(buggy["mutation_order"].unique())
    per_order = max(1, subset_size // len(orders))

    selected = []
    for order in orders:
        candidates = buggy[buggy["mutation_order"] == order]["job_id"].tolist()
        n = min(per_order, len(candidates))
        selected.extend(rng.choice(candidates, size=n, replace=False))

    return selected[:subset_size]


def load_curves(output_dir: Path) -> dict:
    train_losses = np.load(output_dir / "train_losses.npy", allow_pickle=True)
    val_accs_path = output_dir / "val_accs.npy"
    val_accs = np.load(val_accs_path, allow_pickle=True) if val_accs_path.exists() else np.array([])
    test_results_path = output_dir / "test_results.json"
    test_acc = json.load(open(test_results_path))["test_accuracy"] if test_results_path.exists() else None
    return {
        "n_epochs_ran": len(train_losses),
        "train_losses": train_losses,
        "val_accs": val_accs,
        "test_accuracy": test_acc,
    }


def load_buggy_init_curves(program: str, job_ids: list[str]) -> list[dict]:
    curves = []
    for job_id in job_ids:
        d = BUGGY_SAVED_DATA / program / LOSS_FN / f"job_{job_id}"
        try:
            c = load_curves(d)
            c["job_id"] = job_id
            curves.append(c)
        except FileNotFoundError:
            continue
    return curves


def load_random_init_curves(program: str, seeds: list[int]) -> list[dict]:
    curves = []
    for seed in seeds:
        d = RANDOM_INIT_SAVED_DATA / program / f"seed_{seed}" / LOSS_FN / f"job_seed{seed}"
        if not (d / "train_losses.npy").exists():
            continue
        c = load_curves(d)
        c["seed"] = seed
        curves.append(c)
    return curves


def plot_program(program: str, buggy_curves: list[dict], random_curves: list[dict], out_path: Path):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    buggy_color = "#4C72B0"
    random_color = "#C44E52"

    ax = axes[0]
    for i, c in enumerate(buggy_curves):
        ax.plot(np.arange(len(c["train_losses"])), c["train_losses"],
                color=buggy_color, alpha=0.35, linewidth=1,
                label="buggy-init (subset)" if i == 0 else None)
    for i, c in enumerate(random_curves):
        ax.plot(np.arange(len(c["train_losses"])), c["train_losses"],
                color=random_color, alpha=0.6, linewidth=1.3,
                label="random-init" if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title(f"{program}: training loss")
    ax.legend()

    ax = axes[1]
    for i, c in enumerate(buggy_curves):
        if len(c["val_accs"]) == 0:
            continue
        epochs = np.arange(len(c["val_accs"])) * VAL_STEP
        ax.plot(epochs, c["val_accs"] * 100, color=buggy_color, alpha=0.35, linewidth=1,
                label="buggy-init (subset)" if i == 0 else None)
    for i, c in enumerate(random_curves):
        if len(c["val_accs"]) == 0:
            continue
        epochs = np.arange(len(c["val_accs"])) * VAL_STEP
        ax.plot(epochs, c["val_accs"] * 100, color=random_color, alpha=0.6, linewidth=1.3,
                label="random-init" if i == 0 else None)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"{program}: validation accuracy")
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path} (+ .png)")


def summarize(program: str, buggy_curves: list[dict], random_curves: list[dict]) -> dict:
    def stats(curves, key):
        vals = [c[key] for c in curves if c.get(key) is not None]
        if not vals:
            return (float("nan"), float("nan"), 0)
        return (float(np.mean(vals)), float(np.median(vals)), len(vals))

    b_test_mean, b_test_median, b_n = stats(buggy_curves, "test_accuracy")
    r_test_mean, r_test_median, r_n = stats(random_curves, "test_accuracy")
    b_epochs_mean, b_epochs_median, _ = stats(buggy_curves, "n_epochs_ran")
    r_epochs_mean, r_epochs_median, _ = stats(random_curves, "n_epochs_ran")

    return {
        "program": program,
        "buggy_init_n": b_n,
        "buggy_init_test_acc_mean": b_test_mean,
        "buggy_init_test_acc_median": b_test_median,
        "buggy_init_epochs_ran_mean": b_epochs_mean,
        "buggy_init_epochs_ran_median": b_epochs_median,
        "random_init_n": r_n,
        "random_init_test_acc_mean": r_test_mean,
        "random_init_test_acc_median": r_test_median,
        "random_init_epochs_ran_mean": r_epochs_mean,
        "random_init_epochs_ran_median": r_epochs_median,
    }


def make_latex_table(rows: list[dict], out_path: Path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Test accuracy when repairing from a buggy checkpoint vs. from a randomly "
        r"initialized ground-truth model, using the identical repair procedure "
        r"(n=10 seeds for random-init; buggy-init is a stratified subset by mutation order).}",
        r"\label{tab:init-controls}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Program & Buggy-init (n) & Buggy-init acc. & Random-init (n) & Random-init acc. \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['program']} & {row['buggy_init_n']} & {row['buggy_init_test_acc_mean']*100:.1f}\\% & "
            f"{row['random_init_n']} & {row['random_init_test_acc_mean']*100:.1f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-size", type=int, default=15,
                         help="Number of buggy-init mutations to sample per program.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for subset selection.")
    args = parser.parse_args()

    summary_rows = []
    for program in PROGRAMS:
        job_ids = select_buggy_init_subset(program, args.subset_size, seed=args.seed)
        buggy_curves = load_buggy_init_curves(program, job_ids)
        random_curves = load_random_init_curves(program, list(range(N_SEEDS)))

        print(f"{program}: {len(buggy_curves)} buggy-init curves, {len(random_curves)} random-init curves")
        if not random_curves:
            print(f"  (no random-init results yet for {program} - run launch_random_init_baseline.py first)")
            continue

        plot_program(program, buggy_curves, random_curves, PLOTS_DIR / f"init_controls_{program}.pdf")
        summary_rows.append(summarize(program, buggy_curves, random_curves))

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        print(df.to_string(index=False))
        df.to_csv(HERE / "init_controls_summary.csv", index=False)
        make_latex_table(summary_rows, HERE / "init_controls_table.tex")


if __name__ == "__main__":
    main()
