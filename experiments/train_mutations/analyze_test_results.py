#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
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


def load_original_buggy_accuracies():
    """Load original accuracies of buggy models from aggregated mutations file"""
    mutation_path = Path(__file__).parent.parent / "mutation/results/aggregated_mutations.json"
    df = pd.read_json(mutation_path)
    
    original_accuracies = {}
    for _, row in df.iterrows():
        execution_result = row.get("execution_result", {})
        if execution_result.get("status") == "BUGGY_MODEL":
            # The 'accuracy' in execution_result for a BUGGY_MODEL is its accuracy before GBPR
            original_accuracies[row["job_id"]] = execution_result.get("accuracy")
    return original_accuracies


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


def plot_fix_rates(fixed_stats, output_dir=None):
    """Plot the fix rates for different programs, loss functions, and epsilons"""
    # Create plots directory if needed
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Prepare data
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
    for loss_fn in df["Loss Function"].unique():
        loss_fn_data = df[df["Loss Function"] == loss_fn]
        
        fig, ax = plt.subplots(figsize=(12, 6))
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
        
        # Save the plot in loss function subdirectory
        if output_dir:
            loss_fn_dir = plots_dir / loss_fn
            loss_fn_dir.mkdir(exist_ok=True, parents=True)
            output_file = loss_fn_dir / "fix_rates.pdf"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {output_file}")
        plt.close()

def plot_repair_progression(results, epsilons, output_dir=None):
    """Plot the repair progression by mutation order for different epsilon thresholds and loss functions"""
    # Create plots directory if needed
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Get mutation orders and group by loss function
    mutation_orders = load_mutation_orders()
    loss_fn_groups = {}
    for result in results:
        if result["job_id"] in mutation_orders:
            loss_fn = result["loss_function"]
            if loss_fn not in loss_fn_groups:
                loss_fn_groups[loss_fn] = []
            result["mutation_order"] = mutation_orders[result["job_id"]]
            loss_fn_groups[loss_fn].append(result)
    
    # Create separate plot for each loss function
    for loss_fn, loss_fn_results in loss_fn_groups.items():
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort and process results
        sorted_results = sorted(loss_fn_results, key=lambda x: x["mutation_order"])
        order_groups = {}
        for result in sorted_results:
            order = result["mutation_order"]
            if order not in order_groups:
                order_groups[order] = []
            order_groups[order].append(result)
        
        # Calculate and plot fix rates
        fix_rates = {epsilon: [] for epsilon in epsilons}
        orders = sorted(order_groups.keys())
        
        for order in orders:
            group = order_groups[order]
            total = len(group)
            for epsilon in epsilons:
                fixed = sum(1 for r in group if r["test_accuracy"] >= (1.0 - epsilon))
                fix_rates[epsilon].append((fixed / total) * 100 if total > 0 else 0)
        
        for epsilon in epsilons:
            ax.plot(orders, fix_rates[epsilon],
                   label=f'ε={epsilon} (n={len(sorted_results)})',
                   marker='o',
                   markersize=8,
                   linestyle='--')
        
        # Add annotations and styling
        ax.set_xlabel("Mutation Order")
        ax.set_ylabel("Fixed Models (%)")
        ax.set_title("Repair Success Rate", pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        ax.set_xlim(1, max(orders))
        ax.set_ylim(0, 100)
        ax.set_xticks(orders)
        
        # Add total numbers
        for order in orders:
            total = len(order_groups[order])
            ax.annotate(f'n={total}',
                       xy=(order, 100),
                       xytext=(0, 5),
                       textcoords='offset points',
                       ha='center',
                       va='bottom')
        
        plt.tight_layout()
        
        # Save the plot in loss function subdirectory
        if output_dir:
            loss_fn_dir = plots_dir / loss_fn
            loss_fn_dir.mkdir(exist_ok=True, parents=True)
            output_file = loss_fn_dir / "repair_progression.pdf"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {output_file}")
        plt.close()

def plot_repair_progression_per_program(results, epsilons, output_dir=None):
    """Plot the repair progression per program with all programs in one figure"""
    # Create plots directory if needed
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Get mutation orders and group results
    mutation_orders = load_mutation_orders()
    program_loss_fn_groups = {}
    
    for result in results:
        if result["job_id"] in mutation_orders:
            program = result["program_name"]
            loss_fn = result["loss_function"]
            key = (program, loss_fn)
            if key not in program_loss_fn_groups:
                program_loss_fn_groups[key] = []
            result["mutation_order"] = mutation_orders[result["job_id"]]
            program_loss_fn_groups[key].append(result)
    
    # Group by loss function
    loss_fn_groups = {}
    for (program, loss_fn), results in program_loss_fn_groups.items():
        if loss_fn not in loss_fn_groups:
            loss_fn_groups[loss_fn] = {}
        loss_fn_groups[loss_fn][program] = results
    
    # Create plots for each loss function
    for loss_fn, programs in loss_fn_groups.items():
        # Create subplot grid (2x3 or adjusted based on number of programs)
        fig = plt.figure(figsize=(20, 12))
        
        for idx, (program, program_results) in enumerate(sorted(programs.items()), 1):
            ax = fig.add_subplot(2, 3, idx)
            
            # Sort and process results
            sorted_results = sorted(program_results, key=lambda x: x["mutation_order"])
            order_groups = {}
            for result in sorted_results:
                order = result["mutation_order"]
                if order not in order_groups:
                    order_groups[order] = []
                order_groups[order].append(result)
            
            # Calculate and plot fix rates
            fix_rates = {epsilon: [] for epsilon in epsilons}
            orders = sorted(order_groups.keys())
            
            for order in orders:
                group = order_groups[order]
                total = len(group)
                for epsilon in epsilons:
                    fixed = sum(1 for r in group if r["test_accuracy"] >= (1.0 - epsilon))
                    fix_rates[epsilon].append((fixed / total) * 100 if total > 0 else 0)
            
            for epsilon in epsilons:
                ax.plot(orders, fix_rates[epsilon],
                       label=f'ε={epsilon} (n={len(sorted_results)})',
                       marker='o',
                       markersize=6,
                       linestyle='--')
            
            # Add styling and annotations
            ax.set_xlabel("Mutation Order")
            ax.set_ylabel("Fixed Models (%)")
            ax.set_title(f"{program}")
            if idx == 1:  # Only show legend on first subplot
                ax.legend()
            ax.grid(True)
            ax.set_xlim(1, max(orders))
            ax.set_ylim(0, 100)
            ax.set_xticks(orders)
            
            # Add total numbers
            for order in orders:
                total = len(order_groups[order])
                ax.annotate(f'n={total}',
                           xy=(order, 100),
                           xytext=(0, 5),
                           textcoords='offset points',
                           ha='center',
                           va='bottom',
                           fontsize=8)
        
        fig.suptitle("Repair Progression by Program", fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save the plot in loss function subdirectory
        if output_dir:
            loss_fn_dir = plots_dir / loss_fn
            loss_fn_dir.mkdir(exist_ok=True, parents=True)
            output_file = loss_fn_dir / "repair_progression_by_program.pdf"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {output_file}")
        plt.close()

def plot_accuracy_histogram(results, output_dir=None):
    """Plot histogram of test accuracies (before and after GBPR) with summary statistics"""
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    original_buggy_accuracies_map = load_original_buggy_accuracies()

    # Group results by loss function and program
    loss_fn_groups = {} # loss_fn -> list of results for that loss_fn
    program_groups = {} # loss_fn -> program_name -> list of results for that program under that loss_fn
    
    for result in results:
        loss_fn = result["loss_function"]
        program = result["program_name"]
        job_id = result["job_id"]

        # Get original accuracy if available
        original_accuracy = original_buggy_accuracies_map.get(job_id) # Returns None if job_id not found

        if loss_fn not in loss_fn_groups:
            loss_fn_groups[loss_fn] = []
            program_groups[loss_fn] = {}
        if program not in program_groups[loss_fn]:
            program_groups[loss_fn][program] = []
            
        # Store both original and test accuracy
        loss_fn_groups[loss_fn].append({
            "test_accuracy": result["test_accuracy"], 
            "original_accuracy": original_accuracy
        })
        program_groups[loss_fn][program].append({
            "test_accuracy": result["test_accuracy"],
            "original_accuracy": original_accuracy
        })
    
    # Create histograms for each loss function
    for loss_fn, loss_fn_results_list in loss_fn_groups.items():
        loss_fn_dir = plots_dir / loss_fn
        loss_fn_dir.mkdir(exist_ok=True, parents=True)
        
        # Aggregate histogram for the current loss function
        accuracies_after = pd.Series([r["test_accuracy"] * 100 for r in loss_fn_results_list])
        # Filter out None values for original accuracies before converting to Series
        accuracies_before = pd.Series([r["original_accuracy"] * 100 for r in loss_fn_results_list if r["original_accuracy"] is not None])

        plt.figure(figsize=(12, 7))
        plotted_anything_aggregate = False
        
        # Determine common bins for aggregate plot
        combined_data_for_bins_agg = []
        if not accuracies_before.empty:
            combined_data_for_bins_agg.extend(accuracies_before.dropna().tolist())
        if not accuracies_after.empty:
            combined_data_for_bins_agg.extend(accuracies_after.dropna().tolist())

        bins_to_use_agg = 50  # Default

        if combined_data_for_bins_agg:
            min_val = np.min(combined_data_for_bins_agg)
            max_val = np.max(combined_data_for_bins_agg)
            
            if pd.isna(min_val) or pd.isna(max_val): # Should ideally not happen with dropna
                bins_to_use_agg = 50
            elif min_val == max_val:
                # Create a small range around the single value, e.g., +/- 2.5, clamped to [0, 100]
                actual_min = max(0.0, min_val - 2.5)
                actual_max = min(100.0, max_val + 2.5)
                if actual_min >= actual_max: # Ensure a tiny positive range if clamping collapsed it
                    actual_max = actual_min + 1e-6 if actual_min < 100 else 100.0
                    if actual_min == 100.0 : actual_min = 100.0 - 1e-6 # ensure min < max
                bins_to_use_agg = np.linspace(actual_min, actual_max, 50 + 1)
            else:
                bins_to_use_agg = np.linspace(min_val, max_val, 50 + 1)
        
        if not accuracies_before.empty:
            plt.hist(accuracies_before, bins=bins_to_use_agg, edgecolor="black", label='Before GBPR', color='red', alpha=0.7)
            plt.axvline(
                accuracies_before.median(), color="darkred", linestyle="dashed", linewidth=2,
                label=f"Median: {accuracies_before.median():.2f}%"
            )
            plotted_anything_aggregate = True
            
        if not accuracies_after.empty:
            plt.hist(accuracies_after, bins=bins_to_use_agg, edgecolor="black", label='After GBPR', color='green', alpha=0.7)
            plt.axvline(
                accuracies_after.median(), color="darkgreen", linestyle="dashed", linewidth=2,
                label=f"Median: {accuracies_after.median():.2f}%"
            )
            plotted_anything_aggregate = True

        if not plotted_anything_aggregate: # Skip if no data
            plt.close()
            continue
        
        plt.title("Distribution of Test Accuracy (All Programs)")
        plt.xlabel("Test Set Accuracy (%)")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        
        if output_dir:
            output_file = loss_fn_dir / "accuracy_histogram_aggregate.pdf"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {output_file}")
        plt.close()
        
        # Create per-program histograms in a single figure for the current loss function
        current_program_group = program_groups[loss_fn]
        num_programs = len(current_program_group)
        if num_programs == 0:
            continue

        num_cols = min(3, num_programs)
        num_rows = (num_programs + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), squeeze=False)
        axes = axes.flatten()
        
        plotted_program_idx = 0
        for idx, (program, program_results_list) in enumerate(sorted(current_program_group.items())):
            ax = axes[plotted_program_idx] # Use plotted_program_idx to handle cases where some programs have no data
            prog_accuracies_after = pd.Series([r["test_accuracy"] * 100 for r in program_results_list])
            prog_accuracies_before = pd.Series([r["original_accuracy"] * 100 for r in program_results_list if r["original_accuracy"] is not None])
            
            plotted_anything_program = False

            # Determine common bins for this program's plot
            combined_data_for_bins_prog = []
            if not prog_accuracies_before.empty:
                combined_data_for_bins_prog.extend(prog_accuracies_before.dropna().tolist())
            if not prog_accuracies_after.empty:
                combined_data_for_bins_prog.extend(prog_accuracies_after.dropna().tolist())

            bins_to_use_prog = 50  # Default

            if combined_data_for_bins_prog:
                min_val_prog = np.min(combined_data_for_bins_prog)
                max_val_prog = np.max(combined_data_for_bins_prog)

                if pd.isna(min_val_prog) or pd.isna(max_val_prog):
                    bins_to_use_prog = 50
                elif min_val_prog == max_val_prog:
                    actual_min_prog = max(0.0, min_val_prog - 2.5)
                    actual_max_prog = min(100.0, max_val_prog + 2.5)
                    if actual_min_prog >= actual_max_prog:
                        actual_max_prog = actual_min_prog + 1e-6 if actual_min_prog < 100 else 100.0
                        if actual_min_prog == 100.0 : actual_min_prog = 100.0 - 1e-6
                    bins_to_use_prog = np.linspace(actual_min_prog, actual_max_prog, 50 + 1)
                else:
                    bins_to_use_prog = np.linspace(min_val_prog, max_val_prog, 50 + 1)

            if not prog_accuracies_before.empty:
                ax.hist(prog_accuracies_before, bins=bins_to_use_prog, edgecolor="black", label='Before GBPR', color='red', alpha=0.7)
                ax.axvline(
                    prog_accuracies_before.median(), color="darkred", linestyle="dashed", linewidth=1.5,
                    label=f"Median (Before): {prog_accuracies_before.median():.2f}%"
                )
                plotted_anything_program = True

            if not prog_accuracies_after.empty:
                ax.hist(prog_accuracies_after, bins=bins_to_use_prog, edgecolor="black", label='After GBPR', color='green', alpha=0.7)
                ax.axvline(
                    prog_accuracies_after.median(), color="darkgreen", linestyle="dashed", linewidth=1.5,
                    label=f"Median (After): {prog_accuracies_after.median():.2f}%"
                )
                plotted_anything_program = True
            
            if not plotted_anything_program: # Skip if no data for this program
                ax.set_title(f"{program}\n(No accuracy data)")
                ax.set_visible(False) 
                # Do not increment plotted_program_idx here, so the next valid plot uses this ax
                continue

            ax.set_title(f"{program}\n(Models: Before={len(prog_accuracies_before)}, After={len(prog_accuracies_after)})")
            ax.set_xlabel("Accuracy (%)")
            ax.set_ylabel("Count")
            ax.legend(fontsize='small')
            plotted_program_idx += 1
        
        for i in range(plotted_program_idx, len(axes)): # Hide unused subplots
            axes[i].set_visible(False)
        
        if plotted_program_idx == 0: # No programs had data to plot
            plt.close(fig)
            # Continue to stats writing even if no plots generated for by_program
        else:
            fig.suptitle("Distribution of Test Accuracy by Program", y=1.02 if num_rows > 1 else 1.05)
            plt.tight_layout(rect=[0, 0, 1, 0.98 if num_rows > 1 else 0.95]) # Adjust layout to prevent title overlap
            
            if output_dir:
                output_file = loss_fn_dir / "accuracy_histogram_by_program.pdf"
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                print(f"Plot saved to {output_file}")
            plt.close(fig) # Close the figure for per-program plots
            
        # Stats writing (remains unchanged from original logic, but ensure it's outside the conditional plotting)
        if output_dir: # This check was already there for the stats file
            stats_file = loss_fn_dir / "accuracy_stats.txt"
            with open(stats_file, "w") as f:
                f.write(f"=== Statistics for Loss Function: {loss_fn} ===\n")
                
                # Aggregate stats
                f.write("\n--- Aggregate Statistics (All Programs) ---\n")
                if not accuracies_before.empty:
                    f.write("Original Buggy Model Accuracy (Before GBPR):\n")
                    f.write(f"  Number of models: {len(accuracies_before)}\n")
                    f.write(f"  Median accuracy: {accuracies_before.median():.2f}%\n")
                    f.write(f"  Min accuracy: {accuracies_before.min():.2f}%\n")
                    f.write(f"  Max accuracy: {accuracies_before.max():.2f}%\n")
                else:
                    f.write("No 'Before GBPR' accuracy data available for aggregate.\n")

                if not accuracies_after.empty:
                    f.write("Test Accuracy (After GBPR):\n")
                    f.write(f"  Number of models: {len(accuracies_after)}\n")
                    f.write(f"  Median accuracy: {accuracies_after.median():.2f}%\n")
                    f.write(f"  Min accuracy: {accuracies_after.min():.2f}%\n")
                    f.write(f"  Max accuracy: {accuracies_after.max():.2f}%\n")
                else:
                    f.write("No 'After GBPR' accuracy data available for aggregate.\n")

                # Per-program stats
                f.write("\n--- Per-Program Statistics ---\n")
                for program, program_results_list in sorted(current_program_group.items()):
                    # Recalculate these series as they are local to the plotting loop
                    prog_accuracies_after_stats = pd.Series([r["test_accuracy"] * 100 for r in program_results_list])
                    prog_accuracies_before_stats = pd.Series([r["original_accuracy"] * 100 for r in program_results_list if r["original_accuracy"] is not None])
                    
                    f.write(f"\nProgram: {program}\n")
                    if not prog_accuracies_before_stats.empty:
                        f.write("  Original Buggy Model Accuracy (Before GBPR):\n")
                        f.write(f"    Number of models: {len(prog_accuracies_before_stats)}\n")
                        f.write(f"    Median accuracy: {prog_accuracies_before_stats.median():.2f}%\n")
                        f.write(f"    Min accuracy: {prog_accuracies_before_stats.min():.2f}%\n")
                        f.write(f"    Max accuracy: {prog_accuracies_before_stats.max():.2f}%\n")
                    else:
                        f.write("  No 'Before GBPR' accuracy data available for this program.\n")
                    
                    if not prog_accuracies_after_stats.empty:
                        f.write("  Test Accuracy (After GBPR):\n")
                        f.write(f"    Number of models: {len(prog_accuracies_after_stats)}\n")
                        f.write(f"    Median accuracy: {prog_accuracies_after_stats.median():.2f}%\n")
                        f.write(f"    Min accuracy: {prog_accuracies_after_stats.min():.2f}%\n")
                        f.write(f"    Max accuracy: {prog_accuracies_after_stats.max():.2f}%\n")
                    else:
                        f.write("  No 'After GBPR' accuracy data available for this program.\n")

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

    # Generate standard plots
    print("Generating standard plots...")
    plot_fix_rates(fixed_stats, output_path)
    plot_repair_progression(results, epsilon_values, output_path)
    plot_repair_progression_per_program(results, epsilon_values, output_path)
    plot_accuracy_histogram(results, output_path)

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
