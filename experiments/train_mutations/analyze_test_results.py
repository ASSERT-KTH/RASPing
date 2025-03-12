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
    """Load individual test results from the saved_data directory
    
    Directory structure is:
    saved_data/
        program_name/
            loss_function_name/
                job_id/
                    test_results.json
    """
    results = []
    # Find all test_results.json files recursively
    for result_file in Path(saved_data_dir).rglob("test_results.json"):
        try:
            with open(result_file, "r") as f:
                result = json.load(f)
                # Add loss function name from directory path
                # Path structure: saved_data/program/loss_fn/job_id/test_results.json
                result["loss_function"] = result_file.parent.parent.name
                results.append(result)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {result_file}")
            continue
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

    # Group by program name and loss function
    programs = {}  # program -> results
    loss_functions = {}  # loss_fn -> results
    program_loss_functions = {}  # (program, loss_fn) -> results
    # Group by mutation order
    mutation_order_groups = {}  # order -> results
    
    for result in results:
        program = result["program_name"]
        job_id = result["job_id"]
        loss_fn = result["loss_function"]
        
        # Add to program groups
        if program not in programs:
            programs[program] = []
        programs[program].append(result)
        
        # Add to loss function groups
        if loss_fn not in loss_functions:
            loss_functions[loss_fn] = []
        loss_functions[loss_fn].append(result)
        
        # Add to combined program/loss function groups
        key = (program, loss_fn)
        if key not in program_loss_functions:
            program_loss_functions[key] = []
        program_loss_functions[key].append(result)
        
        # Add to mutation order groups
        if job_id in mutation_orders:
            order = mutation_orders[job_id]
            if order not in mutation_order_groups:
                mutation_order_groups[order] = []
            mutation_order_groups[order].append(result)

    # Calculate fixed models for each epsilon
    fixed_stats = {}
    for epsilon in epsilons:
        fixed_stats[epsilon] = {
            "programs": {},
            "loss_functions": {},
            "program_loss_functions": {},
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
        
        # Calculate per loss function stats
        for loss_fn, loss_fn_results in loss_functions.items():
            total = len(loss_fn_results)
            fixed = sum(1 for r in loss_fn_results if r["test_accuracy"] >= (1.0 - epsilon))
            fixed_stats[epsilon]["loss_functions"][loss_fn] = {
                "fixed": fixed,
                "total": total,
                "percentage": (fixed / total) * 100 if total > 0 else 0,
            }
        
        # Calculate per program/loss function combination stats
        for (program, loss_fn), combo_results in program_loss_functions.items():
            total = len(combo_results)
            fixed = sum(1 for r in combo_results if r["test_accuracy"] >= (1.0 - epsilon))
            if program not in fixed_stats[epsilon]["program_loss_functions"]:
                fixed_stats[epsilon]["program_loss_functions"][program] = {}
            fixed_stats[epsilon]["program_loss_functions"][program][loss_fn] = {
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
    """Plot the fix rates for different programs, loss functions, and epsilons"""
    # Prepare data for plotting - combine program and loss function data
    data = []
    for epsilon, stats_dict in fixed_stats.items():
        for program, loss_fn_stats in stats_dict["program_loss_functions"].items():
            for loss_fn, stats in loss_fn_stats.items():
                data.append({
                    "Epsilon": f"ε={epsilon}",
                    "Program": program,
                    "Loss Function": loss_fn,
                    "Fixed (%)": stats["percentage"],
                    "Fixed": stats["fixed"],
                    "Total": stats["total"],
                })

    df = pd.DataFrame(data)

    # Create separate plots for each loss function
    loss_functions = df["Loss Function"].unique()
    n_loss_fns = len(loss_functions)
    fig, axs = plt.subplots(n_loss_fns, 1, figsize=(12, 6 * n_loss_fns))
    if n_loss_fns == 1:
        axs = [axs]

    for idx, loss_fn in enumerate(loss_functions):
        loss_fn_data = df[df["Loss Function"] == loss_fn]
        ax = axs[idx]
        sns.barplot(x="Program", y="Fixed (%)", hue="Epsilon", data=loss_fn_data, ax=ax)

        # Add labels and title
        ax.set_xlabel("Program")
        ax.set_ylabel("Fixed Models (%)")
        ax.set_title(f"Percentage of Fixed Models by Program and Epsilon Threshold\nLoss Function: {loss_fn}")

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%")

        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save the plot if an output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_file}")

    return plt


def plot_repair_progression(results, epsilons, output_file=None):
    """Plot the repair progression by mutation order for different epsilon thresholds and loss functions"""
    # Get mutation orders
    mutation_orders = load_mutation_orders()
    
    # Group results by loss function
    loss_fn_groups = {}
    for result in results:
        if result["job_id"] in mutation_orders:
            loss_fn = result["loss_function"]
            if loss_fn not in loss_fn_groups:
                loss_fn_groups[loss_fn] = []
            result["mutation_order"] = mutation_orders[result["job_id"]]
            loss_fn_groups[loss_fn].append(result)
    
    # Create separate plots for each loss function
    n_loss_fns = len(loss_fn_groups)
    fig, axs = plt.subplots(n_loss_fns, 1, figsize=(12, 6 * n_loss_fns))
    if n_loss_fns == 1:
        axs = [axs]
    
    for idx, (loss_fn, loss_fn_results) in enumerate(loss_fn_groups.items()):
        # Sort results by mutation order
        sorted_results = sorted(loss_fn_results, key=lambda x: x["mutation_order"])
        
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
        
        ax = axs[idx]
        for epsilon in epsilons:
            ax.plot(orders, fix_rates[epsilon],
                   label=f'ε={epsilon} (n={len(sorted_results)})',
                   marker='o',
                   markersize=8,
                   linestyle='--')
        
        ax.set_xlabel("Mutation Order")
        ax.set_ylabel("Fixed Models (%)")
        ax.set_title(f"Repair Success Rate - {loss_fn}", pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
        # Set fixed axis limits
        ax.set_xlim(1, max(orders))
        ax.set_ylim(0, 100)
        ax.set_xticks(orders)
        
        # Add total number of models for each order above the plot
        for order in orders:
            total = len(order_groups[order])
            ax.annotate(f'n={total}',
                       xy=(order, 100),
                       xytext=(0, 5),  # 5 points vertical offset
                       textcoords='offset points',
                       ha='center',
                       va='bottom')
    
    plt.tight_layout()
    
    # Save the plot if an output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
    
    return plt


def plot_repair_progression_per_program(results, epsilons, output_file=None):
    """Plot the repair progression over time for different epsilon thresholds and loss functions, separated by program"""
    # Get mutation orders and group results by program and loss function
    mutation_orders = load_mutation_orders()
    program_loss_fn_groups = {}  # (program, loss_fn) -> results
    
    for result in results:
        if result["job_id"] in mutation_orders:
            program = result["program_name"]
            loss_fn = result["loss_function"]
            key = (program, loss_fn)
            if key not in program_loss_fn_groups:
                program_loss_fn_groups[key] = []
            result["mutation_order"] = mutation_orders[result["job_id"]]
            program_loss_fn_groups[key].append(result)
    
    # Get unique programs and loss functions
    programs = sorted(set(key[0] for key in program_loss_fn_groups.keys()))
    loss_functions = sorted(set(key[1] for key in program_loss_fn_groups.keys()))
    
    # Create subplot grid
    n_programs = len(programs)
    n_loss_fns = len(loss_functions)
    fig, axs = plt.subplots(n_programs, n_loss_fns, 
                           figsize=(6 * n_loss_fns, 4 * n_programs),
                           squeeze=False)
    
    for prog_idx, program in enumerate(programs):
        for loss_idx, loss_fn in enumerate(loss_functions):
            key = (program, loss_fn)
            ax = axs[prog_idx, loss_idx]
            
            if key in program_loss_fn_groups:
                program_results = program_loss_fn_groups[key]
                
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
                
                # Plot repair progression
                for epsilon in epsilons:
                    ax.plot(orders, fix_rates[epsilon],
                           label=f'ε={epsilon} (n={len(sorted_results)})',
                           marker='o',
                           markersize=6,
                           linestyle='--')
                
                # Add total number of models for each order above the plot
                for order in orders:
                    total = len(order_groups[order])
                    ax.annotate(f'n={total}',
                               xy=(order, 100),
                               xytext=(0, 5),
                               textcoords='offset points',
                               ha='center',
                               va='bottom',
                               fontsize=8)
                
                ax.set_xlim(1, max(orders))
                ax.set_ylim(0, 100)
                ax.set_xticks(orders)
            
            # Always set labels and grid
            if prog_idx == n_programs - 1:
                ax.set_xlabel("Mutation Order")
            if loss_idx == 0:
                ax.set_ylabel("Fixed Models (%)")
            
            ax.set_title(f"{program} - {loss_fn}")
            ax.grid(True)
            
            # Only add legend to the first plot
            if prog_idx == 0 and loss_idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
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
