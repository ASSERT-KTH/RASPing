#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import click


def load_results(results_file):
    """Load the aggregated test results"""
    with open(results_file, "r") as f:
        results = json.load(f)
    return results


def analyze_fixed_models(results, epsilons=None):
    """Analyze how many models are fixed based on different epsilon thresholds"""
    if epsilons is None:
        epsilons = [0.0, 0.001, 0.01]

    # Group by program name
    programs = {}
    for result in results:
        program = result["program_name"]
        if program not in programs:
            programs[program] = []
        programs[program].append(result)

    # Calculate fixed models for each epsilon and program
    fixed_stats = {}
    for epsilon in epsilons:
        fixed_stats[epsilon] = {}
        for program, program_results in programs.items():
            total = len(program_results)
            fixed = sum(
                1 for r in program_results if r["test_accuracy"] >= (1.0 - epsilon)
            )
            fixed_stats[epsilon][program] = {
                "fixed": fixed,
                "total": total,
                "percentage": (fixed / total) * 100 if total > 0 else 0,
            }

    return fixed_stats


def plot_fix_rates(fixed_stats, output_file=None):
    """Plot the fix rates for different programs and epsilons"""
    # Prepare data for plotting
    data = []
    for epsilon, programs in fixed_stats.items():
        for program, stats in programs.items():
            data.append(
                {
                    "Epsilon": f"ε={epsilon}",
                    "Program": program,
                    "Fixed (%)": stats["percentage"],
                    "Fixed": stats["fixed"],
                    "Total": stats["total"],
                }
            )

    df = pd.DataFrame(data)

    # Create the plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x="Program", y="Fixed (%)", hue="Epsilon", data=df)

    # Add labels and title
    plt.xlabel("Program")
    plt.ylabel("Fixed Models (%)")
    plt.title("Percentage of Fixed Models by Program and Epsilon Threshold")

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%")

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot if an output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_file}")

    return plt


@click.command()
@click.option(
    "--results-file",
    default="test_results_aggregated.json",
    help="Path to the aggregated test results JSON file",
)
@click.option(
    "--output-dir",
    default="analysis_outputs",
    help="Directory to save analysis outputs",
)
@click.option(
    "--epsilons",
    default="0.0,0.001,0.01",
    help="Comma-separated list of epsilon values",
)
def main(results_file, output_dir, epsilons):
    """Analyze and visualize test results from trained mutation models"""
    # Parse epsilons
    epsilon_values = [float(e) for e in epsilons.split(",")]

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load results
    print(f"Loading results from {results_file}...")
    results = load_results(results_file)
    print(f"Loaded {len(results)} results")

    # Analyze fixed models
    print("Analyzing fixed models...")
    fixed_stats = analyze_fixed_models(results, epsilon_values)

    # Save fixed stats
    fixed_stats_file = output_path / "fixed_stats.json"
    with open(fixed_stats_file, "w") as f:
        json.dump(fixed_stats, f, indent=2)
    print(f"Fixed model statistics saved to {fixed_stats_file}")

    # Plot fix rates
    print("Plotting fix rates...")
    fix_rates_plot = plot_fix_rates(fixed_stats, output_path / "fix_rates.png")
    plt.close()

    # Generate summary table
    summary_file = output_path / "summary.md"
    with open(summary_file, "w") as f:
        f.write("# Test Results Summary\n\n")
        f.write("## Fix Rates by Epsilon\n\n")

        for epsilon in epsilon_values:
            f.write(f"### Epsilon = {epsilon}\n\n")
            f.write("| Program | Fixed | Total | Percentage |\n")
            f.write("|---------|-------|-------|------------|\n")

            for program, stats in fixed_stats[epsilon].items():
                f.write(
                    f"| {program} | {stats['fixed']} | {stats['total']} | {stats['percentage']:.2f}% |\n"
                )

            f.write("\n")

    print(f"Summary saved to {summary_file}")
    print(f"All analysis outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
