#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import glob
import click


def load_results(saved_data_dir):
    """Load individual test results from the saved_data directory"""
    results = []
    # Find all test_results.json files recursively
    for result_file in Path(saved_data_dir).rglob("test_results.json"):
        with open(result_file, "r") as f:
            result = json.load(f)
            results.append(result)
    return results


def load_mutation_orders():
    """Load mutation orders from aggregated mutations file"""
    mutation_path = Path(__file__).parent.parent / "mutation/results/aggregated_mutations.json"
    df = pd.read_json(mutation_path)
    
    # Create a mapping of job_id to mutation_order
    mutation_orders = {}
    for _, row in df.iterrows():
        if row["execution_result"].get("status") == "BUGGY_MODEL":
            mutation_orders[row["job_id"]] = row["mutation_order"]
    return mutation_orders


def analyze_fixed_models(results, epsilons=None):
    """Analyze how many models are fixed based on different epsilon thresholds"""
    if epsilons is None:
        epsilons = [0.0, 0.001, 0.01]

    # Get mutation orders
    mutation_orders = load_mutation_orders()

    # Group by program name
    programs = {}
    # Group by mutation order
    mutation_order_groups = {}
    
    for result in results:
        program = result["program_name"]
        job_id = result["job_id"]
        
        # Add to program groups
        if program not in programs:
            programs[program] = []
        programs[program].append(result)
        
        # Add to mutation order groups
        if job_id in mutation_orders:
            order = mutation_orders[job_id]
            if order not in mutation_order_groups:
                mutation_order_groups[order] = []
            mutation_order_groups[order].append(result)

    # Calculate fixed models for each epsilon and program
    fixed_stats = {}
    for epsilon in epsilons:
        fixed_stats[epsilon] = {
            "programs": {},
            "mutation_orders": {}
        }
        # Calculate per program stats
        for program, program_results in programs.items():
            total = len(program_results)
            fixed = sum(1 for r in program_results if r["test_accuracy"] >= (1.0 - epsilon))
            fixed_stats[epsilon]["programs"][program] = {
                "fixed": fixed,
                "total": total,
                "percentage": (fixed / total) * 100 if total > 0 else 0,
            }
        
        # Calculate per mutation order stats
        for order, order_results in mutation_order_groups.items():
            total = len(order_results)
            fixed = sum(1 for r in order_results if r["test_accuracy"] >= (1.0 - epsilon))
            fixed_stats[epsilon]["mutation_orders"][order] = {
                "fixed": fixed,
                "total": total,
                "percentage": (fixed / total) * 100 if total > 0 else 0,
            }

    return fixed_stats


def plot_fix_rates(fixed_stats, output_file=None):
    """Plot the fix rates for different programs and epsilons"""
    # Prepare data for plotting
    data = []
    for epsilon, stats_dict in fixed_stats.items():
        for program, stats in stats_dict["programs"].items():
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


def plot_repair_progression(results, epsilons, output_file=None):
    """Plot the repair progression by mutation order for different epsilon thresholds"""
    # Get mutation orders
    mutation_orders = load_mutation_orders()
    
    # Prepare results with mutation orders
    results_with_orders = []
    for result in results:
        job_id = result["job_id"]
        if job_id in mutation_orders:
            result["mutation_order"] = mutation_orders[job_id]
            results_with_orders.append(result)
    
    # Sort results by mutation order
    sorted_results = sorted(results_with_orders, key=lambda x: x["mutation_order"])
    
    # Group results by mutation order
    order_groups = {}
    for result in sorted_results:
        order = result["mutation_order"]
        if order not in order_groups:
            order_groups[order] = []
        order_groups[order].append(result)
    
    # Calculate fix rates for each mutation order and epsilon
    fix_rates = {epsilon: [] for epsilon in epsilons}
    orders = sorted(order_groups.keys())
    
    for order in orders:
        group = order_groups[order]
        total = len(group)
        for epsilon in epsilons:
            fixed = sum(1 for r in group if r["test_accuracy"] >= (1.0 - epsilon))
            fix_rates[epsilon].append((fixed / total) * 100 if total > 0 else 0)

    # Create the plot
    plt.figure(figsize=(12, 8))
    for epsilon in epsilons:
        plt.plot(orders, fix_rates[epsilon], 
                label=f'ε={epsilon} (n={len(sorted_results)})',
                marker='o',
                markersize=8,
                linestyle='--')

    plt.xlabel("Mutation Order")
    plt.ylabel("Fixed Models (%)")
    # Add padding to title
    plt.title("Repair Success Rate by Mutation Order", pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Set fixed axis limits
    plt.xlim(1, max(orders))
    plt.ylim(0, 100)
    plt.xticks(orders)

    # Add total number of models for each order above the plot
    for order in orders:
        total = len(order_groups[order])
        plt.annotate(f'n={total}', 
                    xy=(order, 100),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords='offset points',
                    ha='center',
                    va='bottom')

    # Adjust layout to accommodate legend and annotations
    plt.tight_layout()

    # Save the plot if an output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_file}")

    return plt


def plot_repair_progression_per_program(results, epsilons, output_file=None):
    """Plot the repair progression over time for different epsilon thresholds, separated by program"""
    # Get mutation orders and group results by program
    mutation_orders = load_mutation_orders()
    programs = {}
    
    for result in results:
        if result["job_id"] in mutation_orders:
            program = result["program_name"]
            if program not in programs:
                programs[program] = []
            result["mutation_order"] = mutation_orders[result["job_id"]]
            programs[program].append(result)
    
    # Create subplot grid based on number of programs
    n_programs = len(programs)
    n_cols = 2  # You can adjust this if needed
    n_rows = (n_programs + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for idx, (program, program_results) in enumerate(programs.items(), 1):
        # Sort results by mutation order
        sorted_results = sorted(program_results, key=lambda x: x["mutation_order"])
        
        # Group results by mutation order
        order_groups = {}
        for result in sorted_results:
            order = result["mutation_order"]
            if order not in order_groups:
                order_groups[order] = []
            order_groups[order].append(result)
        
        # Calculate fix rates for each mutation order and epsilon
        fix_rates = {epsilon: [] for epsilon in epsilons}
        orders = sorted(order_groups.keys())
        
        for order in orders:
            group = order_groups[order]
            total = len(group)
            for epsilon in epsilons:
                fixed = sum(1 for r in group if r["test_accuracy"] >= (1.0 - epsilon))
                fix_rates[epsilon].append((fixed / total) * 100 if total > 0 else 0)
        
        # Create subplot
        plt.subplot(n_rows, n_cols, idx)
        for epsilon in epsilons:
            plt.plot(orders, fix_rates[epsilon], 
                    label=f'ε={epsilon} (n={len(sorted_results)})',
                    marker='o',
                    markersize=8,
                    linestyle='--')
        
        plt.xlabel("Mutation Order")
        plt.ylabel("Fixed Models (%)")
        # Add padding to title
        plt.title(f"Repair Success Rate - {program}", pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        # Set fixed axis limits
        plt.xlim(1, max(orders))
        plt.ylim(0, 100)
        plt.xticks(orders)

        # Add total number of models for each order above the plot
        for order in orders:
            total = len(order_groups[order])
            plt.annotate(f'n={total}', 
                        xy=(order, 100),
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords='offset points',
                        ha='center',
                        va='bottom')
    
    # Adjust layout to accommodate legends and annotations
    plt.tight_layout()
    
    # Save the plot if an output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
    
    return plt


@click.command()
@click.option(
    "--saved-data-dir",
    default="saved_data",
    help="Directory containing test results",
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
def main(saved_data_dir, output_dir, epsilons):
    """Analyze and visualize test results from trained mutation models"""
    # Parse epsilons
    epsilon_values = [float(e) for e in epsilons.split(",")]

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load results
    print(f"Loading results from {saved_data_dir}...")
    results = load_results(saved_data_dir)
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

    # Plot repair progression
    print("Plotting repair progression...")
    progression_plot = plot_repair_progression(results, epsilon_values, output_path / "repair_progression.png")
    plt.close()

    # Plot repair progression per program
    print("Plotting repair progression per program...")
    progression_per_program_plot = plot_repair_progression_per_program(results, epsilon_values, output_path / "repair_progression_per_program.png")
    plt.close()

    # Generate summary table
    summary_file = output_path / "summary.md"
    with open(summary_file, "w") as f:
        f.write("# Test Results Summary\n\n")
        
        # First write fix rates by program
        f.write("## Fix Rates by Program\n\n")
        for epsilon in epsilon_values:
            f.write(f"### Epsilon = {epsilon}\n\n")
            f.write("| Program | Fixed | Total | Percentage |\n")
            f.write("|---------|-------|-------|------------|\n")

            for program, stats in fixed_stats[epsilon]["programs"].items():
                f.write(
                    f"| {program} | {stats['fixed']} | {stats['total']} | {stats['percentage']:.2f}% |\n"
                )
            f.write("\n")
            
        # Then write fix rates by mutation order
        f.write("## Fix Rates by Number of Mutations\n\n")
        for epsilon in epsilon_values:
            f.write(f"### Epsilon = {epsilon}\n\n")
            f.write("| Number of Mutations | Fixed | Total | Percentage |\n")
            f.write("|-------------------|-------|-------|------------|\n")

            # Sort by mutation order number
            sorted_orders = sorted(fixed_stats[epsilon]["mutation_orders"].keys())
            for order in sorted_orders:
                stats = fixed_stats[epsilon]["mutation_orders"][order]
                f.write(
                    f"| {order} | {stats['fixed']} | {stats['total']} | {stats['percentage']:.2f}% |\n"
                )
            f.write("\n")

    print(f"Summary saved to {summary_file}")
    print(f"All analysis outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
