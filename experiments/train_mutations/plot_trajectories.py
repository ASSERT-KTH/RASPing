import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from sklearn.decomposition import PCA
import haiku as hk
from pathlib import Path
import pickle
import click
import sys
import os

# Add parent directory to path for imports
module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.loss import (
    cross_entropy_loss,
    cross_entropy_loss_smoothed_accuracy,
    cross_entropy_loss_with_perfect_sequence,
)
from src.trainer import Trainer
from experiments.mutation.load_mutations import load_buggy_models
from src.functions import load_dataset, encodeAndPadData

# Dictionary of loss functions matching the one in train_mutations.py
LOSS_FUNCTIONS = {
    "cross_entropy_loss": cross_entropy_loss,
    "cross_entropy_loss_smoothed_accuracy": cross_entropy_loss_smoothed_accuracy,
    "cross_entropy_loss_with_perfect_sequence": cross_entropy_loss_with_perfect_sequence,
}


def compute_loss_landscape(trajectory, program_name, loss_function_name, job_id, grid_size=100, grid_range_factor=0.2):
    """Compute the loss landscape around the trajectory in PCA space
    
    Args:
        trajectory: List of (step, params, loss) tuples
        program_name: Name of the program model
        loss_function_name: Name of the loss function
        job_id: Job ID for the specific model
        grid_size: Number of points in each dimension of the grid
        grid_range_factor: How much to extend beyond the min/max of trajectory points
        
    Returns:
        Tuple of (X_grid, Y_grid, Z_grid, pca_model, param_matrix)
    """
    print(f"Computing loss landscape for {program_name}/{loss_function_name}/{job_id}...")
    job_id = job_id.replace("job_", "")
    
    # 1. Extract parameters and flatten them
    params_list = [item[1] for item in trajectory]
    flattened_params = []
    
    for params in params_list:
        leaves = jax.tree_util.tree_leaves(params)
        flattened_params.append(jnp.concatenate([jnp.ravel(leaf) for leaf in leaves]))
    
    param_matrix = jnp.stack(flattened_params)
    
    # 2. Apply PCA to get the principal components
    pca = PCA(n_components=2)
    projected_params = pca.fit_transform(param_matrix)
    
    # 3. Create a grid in PCA space
    min_x, max_x = projected_params[:, 0].min(), projected_params[:, 0].max()
    min_y, max_y = projected_params[:, 1].min(), projected_params[:, 1].max()
    
    # Extend the grid beyond the trajectory points
    range_x = max_x - min_x
    range_y = max_y - min_y
    min_x -= range_x * grid_range_factor
    max_x += range_x * grid_range_factor
    min_y -= range_y * grid_range_factor
    max_y += range_y * grid_range_factor
    
    x_grid = np.linspace(min_x, max_x, grid_size)
    y_grid = np.linspace(min_y, max_y, grid_size)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    Z_grid = np.zeros_like(X_grid)
    
    # 4. Load the model and set up loss function
    try:
        # Use same mapping as in train_mutations.py
        program_name_key = program_name
        if program_name == "most_freq":
            program_name_key = "most-freq"
        elif program_name == "shuffle_dyck":
            program_name_key = "shuffle_dyck1"
            
        # Load the buggy model
        model = load_buggy_models(
            max_length=10,  # Default, should be same as training
            program_name=program_name,
            job_id=job_id
        )[job_id]
        
        # Load dataset
        data_path = f"{Path(__file__).parent.resolve()}/../../data/"
        val_dataset = load_dataset(data_path, program_name_key, split_name="val")
        
        # Encode the dataset
        X_val, Y_val = encodeAndPadData(
            val_dataset, model.raspFunction, model.inputs, 10  # Default max_len
        )
        
        # Get loss function
        if loss_function_name not in LOSS_FUNCTIONS:
            print(f"Warning: Loss function {loss_function_name} not found, using default")
            loss_fn = cross_entropy_loss
        else:
            loss_fn = LOSS_FUNCTIONS[loss_function_name]
        
        # 5. Set up forward pass for loss computation
        padToken = model.model.input_encoder.encoding_map["compiler_pad"]
        
        # We need to define a function to compute loss at a given parameter vector
        @jax.jit
        def compute_loss_at_params(params):
            # Reconstruct the model's parameter structure
            reconstructed_params = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params_list[0]), 
                                                               jax.tree_util.tree_leaves(params))
            
            # Define forward pass
            def forward(x):
                compiled_model = model.model.get_compiled_model()
                compiled_model.use_unembed_argmax = False
                compiled_model.pad_token = padToken
                return compiled_model(x, use_dropout=False)
       
            return loss_fn.apply(reconstructed_params, X_val, Y_val, padToken, forward=forward)
        
        # 6. Compute loss for each grid point
        print(f"Computing loss at {grid_size}x{grid_size} grid points...")
        for i in range(grid_size):
            for j in range(grid_size):
                # Convert PCA grid point back to parameter space
                pca_point = np.array([X_grid[i, j], Y_grid[i, j]])
                # Inverse transform to get parameters in original space
                param_vector = pca.inverse_transform([pca_point])[0]
                
                try:
                    # Create full parameter vector structure to match original
                    param_structure = jax.tree_util.tree_structure(params_list[0])
                    param_leaves = jax.tree_util.tree_leaves(params_list[0])
                    
                    # Check if the param_vector length matches the flattened params
                    if len(param_vector) != sum(leaf.size for leaf in param_leaves):
                        print(f"Warning: Parameter vector dimension mismatch at grid point ({i},{j})")
                        Z_grid[i, j] = np.nan
                        continue
                    
                    # Reshape into nested parameter structure
                    reconstructed_params = reconstruct_params(param_vector, params_list[0])
                    
                    # Compute loss at this parameter point
                    try:
                        loss_value = compute_loss_at_params(reconstructed_params)
                        Z_grid[i, j] = loss_value
                    except Exception as e:
                        print(f"Error computing loss at grid point ({i},{j}): {e}")
                        Z_grid[i, j] = np.nan
                        
                except Exception as e:
                    print(f"Error at grid point ({i},{j}): {e}")
                    Z_grid[i, j] = np.nan
                    
    except Exception as e:
        print(f"Error setting up loss computation: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Mask Z_grid values above ceil of initial loss
    initial_loss = trajectory[0][2]
    loss_threshold = 2* np.ceil(initial_loss)
    Z_grid[Z_grid > loss_threshold] = np.nan
    
    return X_grid, Y_grid, Z_grid, pca, param_matrix


def reconstruct_params(flat_params, template_params):
    """Reconstruct a parameter tree structure from a flattened parameter vector.
    
    Args:
        flat_params: Flattened parameter vector
        template_params: Template parameter structure to match
        
    Returns:
        Reconstructed parameter tree
    """
    # Get the leaves and structure of the template
    flat_leaves = jax.tree_util.tree_leaves(template_params)
    treedef = jax.tree_util.tree_structure(template_params)
    
    # Calculate sizes and create new leaves
    sizes = [np.prod(leaf.shape) for leaf in flat_leaves]
    new_leaves = []
    
    idx = 0
    for size, leaf in zip(sizes, flat_leaves):
        # Extract portion of flat_params for this leaf
        param_slice = flat_params[idx:idx + size]
        # Reshape to match original leaf shape
        new_leaf = param_slice.reshape(leaf.shape)
        new_leaves.append(new_leaf)
        idx += size
    
    # Reconstruct the parameter tree
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def plot_loss_landscape_trajectory(trajectories, output_dir):
    """Perform PCA on parameter trajectories and plot the first 2 components vs loss."""
    plots_dir = Path(output_dir) / "plots" / "loss_landscape_trajectories"
    plots_dir.mkdir(exist_ok=True, parents=True)

    print(f"Generating Loss Landscape Trajectory plots for {len(trajectories)} models...")
    processed_count = 0

    original_usetex_setting = plt.rcParams['text.usetex']
    original_latex_preamble = plt.rcParams['text.latex.preamble']
    
    plt.rcParams['text.usetex'] = True
    # Add xcolor package for \textcolor command
    if '\\usepackage{xcolor}' not in original_latex_preamble:
        plt.rcParams['text.latex.preamble'] = original_latex_preamble + '\n\\usepackage{xcolor}'

    def escape_latex(s):
        """Escape special LaTeX characters in a string."""
        return str(s).replace('\\', '\\textbackslash{}') \
                  .replace('{', '\\{') \
                  .replace('}', '\\}') \
                  .replace('#', '\\#') \
                  .replace('$', '\\$') \
                  .replace('%', '\\%') \
                  .replace('&', '\\&') \
                  .replace('_', '\\_') \
                  .replace('^', '\\^') \
                  .replace('~', '\\textasciitilde{}')

    for key, trajectory in trajectories.items():
        program_name, loss_function, job_id = key
        
        try:
            # 1. Extract parameters, steps and losses
            steps = np.array([item[0] for item in trajectory])
            params_list = [item[1] for item in trajectory]
            losses = np.array([item[2] for item in trajectory]).astype(float)

            # 2. Flatten parameters
            if not all(isinstance(p, (dict, hk.Params)) for p in params_list):
                 print(f"Warning: Skipping {key} - trajectory contains non-parameter objects.")
                 continue

            flattened_params = []
            for params in params_list:
                leaves = jax.tree_util.tree_leaves(params)
                if not all(isinstance(leaf, (jnp.ndarray, np.ndarray)) for leaf in leaves):
                    print(f"Warning: Skipping {key} - non-array leaf found in parameters at step {steps[len(flattened_params)]}.")
                    flattened_params = None
                    break 
                if leaves:
                    flattened_params.append(jnp.concatenate([jnp.ravel(leaf) for leaf in leaves]))
                else:
                     flattened_params.append(jnp.array([]))
            
            if flattened_params is None:
                 continue
                 
            # Ensure all flattened vectors have the same dimension
            if len(set(p.shape[0] for p in flattened_params)) > 1:
                print(f"Warning: Skipping {key} - inconsistent parameter vector lengths.")
                continue
            
            param_matrix = jnp.stack(flattened_params)

            # Handle cases with insufficient data points for PCA
            n_samples, n_features = param_matrix.shape
            if n_samples < 3:
                print(f"Warning: Skipping {key} - insufficient data points (samples={n_samples}) for PCA.")
                continue
            if n_features < 2:
                print(f"Warning: Skipping {key} - insufficient parameter features (features={n_features}) for 2D PCA.")
                continue

            # 3. Apply PCA (reduce to 2 components)
            pca = PCA(n_components=2)
            projected_params = pca.fit_transform(param_matrix)
            trajectory_pc2d = projected_params

            # 4. Create two separate figures - one for surface, one for contour
            # Surface plot
            fig1 = plt.figure(figsize=(12, 8))
            ax1 = fig1.add_subplot(111, projection='3d')
            
            # Contour plot
            fig2 = plt.figure(figsize=(12, 8))
            ax2 = fig2.add_subplot(111)
            
            # Compute and plot loss landscape grid
            landscape_result = compute_loss_landscape(trajectory, program_name, loss_function, job_id)
            
            if landscape_result is not None:
                X_grid, Y_grid, Z_grid, _, _ = landscape_result
                
                # Apply filtering/clipping to handle extreme values
                Z_grid_plot = np.copy(Z_grid)
                # Do NOT fill NaNs with mean or any value
                p95 = np.nanpercentile(Z_grid_plot, 95)
                Z_grid_plot = np.clip(Z_grid_plot, None, p95 * 1.5)
                
                # Ensure NaNs are present for transparency
                # (already present from masking step)
                cmap = plt.cm.viridis.copy()
                cmap.set_bad(color=(0, 0, 0, 0))  # Transparent for masked
                
                # Plot surface
                surf = ax1.plot_surface(X_grid, Y_grid, Z_grid_plot, 
                                     cmap=cmap, alpha=1.0, linewidth=0,
                                     antialiased=True, zorder=1)
                fig1.colorbar(surf, ax=ax1, shrink=0.6, aspect=20, label='Loss Value')
                
                # Plot contour
                contour = ax2.contour(X_grid, Y_grid, Z_grid_plot, 
                                    levels=20, colors='k', alpha=0.7)
                contourf = ax2.contourf(X_grid, Y_grid, Z_grid_plot, 
                                      levels=20, cmap=cmap, alpha=1.0)
                fig2.colorbar(contourf, ax=ax2, shrink=0.6, aspect=20, label='Loss Value')

            # 5. Select points for showing examples (start, middle, end)
            example_indices = [0, len(trajectory)//2, -1]  # Start, middle, end
            example_markers = ['X', 'o', '*']
            example_labels = ['Start', 'Mid', 'End'] # Simplified labels
            
            # 6. Load model and generate examples
            try:
                # Use same mapping as in train_mutations.py
                program_name_key = program_name
                if program_name == "most_freq":
                    program_name_key = "most-freq"
                elif program_name == "shuffle_dyck":
                    program_name_key = "shuffle_dyck1"
                    
                # Load the model
                job_id_clean = job_id.replace("job_", "")
                model = load_buggy_models(
                    max_length=10,
                    program_name=program_name,
                    job_id=job_id_clean
                )[job_id_clean]
                
                # Load dataset for examples
                data_path = f"{Path(__file__).parent.resolve()}/../../data/"
                val_dataset = load_dataset(data_path, program_name_key, split_name="val")
                
                # Get a few example inputs
                example_inputs = val_dataset[:2]
                
                # Plot trajectory and examples on both plots
                for plot_idx, ax in enumerate([ax1, ax2]):
                    # Plot full trajectory
                    if plot_idx == 0:  # 3D surface plot
                        ax.plot(trajectory_pc2d[:, 0], trajectory_pc2d[:, 1], losses, 
                               color='r', linestyle='-', linewidth=2, label='Optimization Path', zorder=2)
                    else:  # 2D contour plot
                        ax.plot(trajectory_pc2d[:, 0], trajectory_pc2d[:, 1], 
                               color='r', linestyle='-', linewidth=2, label='Optimization Path', zorder=2)
                    
                    # Plot example points and add textboxes
                    for idx, (i, marker, label) in enumerate(zip(example_indices, example_markers, example_labels)):
                        # Plot point
                        if plot_idx == 0:  # 3D surface plot
                            ax.scatter(trajectory_pc2d[i, 0], trajectory_pc2d[i, 1], losses[i],
                                       s=150, marker=marker, label=label, depthshade=False, zorder=3)
                            
                            # Add example textbox for surface plot
                            # Set model parameters to this point
                            model.model.params = params_list[i]
                            
                            # Generate examples
                            example_text = f"{escape_latex(label)}:\n"
                            for input_seq, true_output in example_inputs:
                                # Get model output
                                output_seq = model.apply(input_seq)
                                # Filter "BOS" tokens for display and format strings
                                # Escape tokens for LaTeX
                                clean_input_tokens = [escape_latex(str(token)) for token in input_seq if str(token).upper() != "BOS"]
                                # Original tokens for comparison logic, escaped tokens for display
                                original_clean_output_tokens = [str(token) for token in output_seq if str(token).upper() != "BOS"]
                                original_clean_true_output_tokens = [str(token) for token in true_output if str(token).upper() != "BOS"]
                                
                                display_input_seq = " ".join(clean_input_tokens)
                                
                                colored_output_parts = []
                                for k_token_idx, output_token_k_orig in enumerate(original_clean_output_tokens):
                                    output_token_k_escaped = escape_latex(output_token_k_orig)
                                    if k_token_idx < len(original_clean_true_output_tokens) and output_token_k_orig == original_clean_true_output_tokens[k_token_idx]:
                                        colored_output_parts.append("\\textcolor{green}{" + output_token_k_escaped + "}")
                                    else:
                                        colored_output_parts.append("\\textcolor{red}{" + output_token_k_escaped + "}")
                                
                                display_colored_output_seq = " ".join(colored_output_parts)
                                if not display_colored_output_seq: # Handle empty output
                                    display_colored_output_seq = "\\textit{(empty output)}"
                                
                                example_text += f"{program_name}({display_input_seq}) = {display_colored_output_seq}\n"
                            
                            example_text = example_text.rstrip("\n") # Remove trailing newline

                            # Create text box with 3D positioning
                            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                            # Position text boxes on alternating sides
                            if idx % 2 == 0:
                                x_offset = 0.05
                                alignment = 'left'
                            else:
                                x_offset = -0.05
                                alignment = 'right'
                            
                            # For 3D plot, we need to position the text box in 3D space
                            x_range = max(X_grid.flatten()) - min(X_grid.flatten())
                            z_range = max(Z_grid_plot.flatten()) - min(Z_grid_plot.flatten())
                            text_x = trajectory_pc2d[i, 0] + x_offset * x_range
                            text_y = trajectory_pc2d[i, 1]
                            text_z = losses[i] + 0.1 * z_range  # Offset in z direction
                            
                            ax.text(text_x, text_y, text_z,
                                  example_text,
                                  bbox=props,
                                  ha=alignment,
                                  va='bottom',
                                  fontsize=8,
                                  zorder=4)
                            
                            # Add a line connecting point to text
                            ax.plot([trajectory_pc2d[i, 0], text_x],
                                  [trajectory_pc2d[i, 1], text_y],
                                  [losses[i], text_z],
                                  linestyle=':', # Changed linestyle
                                  alpha=0.3,     # Changed alpha
                                  zorder=3)
                            
                        else:  # 2D contour plot
                            ax.scatter(trajectory_pc2d[i, 0], trajectory_pc2d[i, 1],
                                       s=150, marker=marker, label=label, zorder=3)
                            
                            # Add example textbox (only on contour plot)
                            # Set model parameters to this point
                            model.model.params = params_list[i]
                            
                            # Generate examples
                            example_text = f"{escape_latex(label)}:\n"
                            for input_seq, true_output in example_inputs:
                                # Get model output
                                output_seq = model.apply(input_seq)
                                # Filter "BOS" tokens for display and format strings
                                # Escape tokens for LaTeX
                                clean_input_tokens = [escape_latex(str(token)) for token in input_seq if str(token).upper() != "BOS"]
                                # Original tokens for comparison logic, escaped tokens for display
                                original_clean_output_tokens = [str(token) for token in output_seq if str(token).upper() != "BOS"]
                                original_clean_true_output_tokens = [str(token) for token in true_output if str(token).upper() != "BOS"]

                                display_input_seq = " ".join(clean_input_tokens)

                                colored_output_parts = []
                                for k_token_idx, output_token_k_orig in enumerate(original_clean_output_tokens):
                                    output_token_k_escaped = escape_latex(output_token_k_orig)
                                    if k_token_idx < len(original_clean_true_output_tokens) and output_token_k_orig == original_clean_true_output_tokens[k_token_idx]:
                                        colored_output_parts.append("\\textcolor{green}{" + output_token_k_escaped + "}")
                                    else:
                                        colored_output_parts.append("\\textcolor{red}{" + output_token_k_escaped + "}")
                                
                                display_colored_output_seq = " ".join(colored_output_parts)
                                if not display_colored_output_seq: # Handle empty output
                                    display_colored_output_seq = "\\textit{(empty output)}"

                                example_text += f"{program_name}({display_input_seq}) = {display_colored_output_seq}\n"
                            
                            example_text = example_text.rstrip("\n") # Remove trailing newline
                            
                            # Create text box
                            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                            # Position text boxes on alternating sides
                            if idx % 2 == 0:
                                x_offset = 0.05
                                alignment = 'left'
                            else:
                                x_offset = -0.05
                                alignment = 'right'
                            
                            ax.annotate(example_text,
                                      xy=(trajectory_pc2d[i, 0], trajectory_pc2d[i, 1]),
                                      xytext=(trajectory_pc2d[i, 0] + x_offset * (max(X_grid.flatten()) - min(X_grid.flatten())),
                                             trajectory_pc2d[i, 1]),
                                      bbox=props,
                                      ha=alignment,
                                      va='center',
                                      fontsize=8,
                                      arrowprops=dict(arrowstyle="-"))
                
                # Add labels and title
                if plot_idx == 0:  # 3D surface plot
                    ax1.set_xlabel('Principal Component 1')
                    ax1.set_ylabel('Principal Component 2')
                    ax1.set_zlabel('Loss')
                    ax1.set_title('Loss Landscape Surface Plot\n' + f'{program_name} - {loss_function} - {job_id}')
                    ax1.legend()
                else:  # 2D contour plot
                    ax2.set_xlabel('Principal Component 1')
                    ax2.set_ylabel('Principal Component 2')
                    ax2.set_title('Loss Landscape Contour Plot with Examples\n' + f'{program_name} - {loss_function} - {job_id}')
                    ax2.legend()
                
            except Exception as e:
                print(f"Error generating examples for {key}: {e}")
                import traceback
                traceback.print_exc()
            
            # 7. Save plots
            plot_subdir = plots_dir / program_name / loss_function
            plot_subdir.mkdir(exist_ok=True, parents=True)
            
            # Save surface plot
            surface_file = plot_subdir / f"{job_id}_loss_landscape_surface.png"
            fig1.savefig(surface_file, dpi=300, bbox_inches="tight")
            plt.close(fig1)
            
            # Save contour plot with examples
            contour_file = plot_subdir / f"{job_id}_loss_landscape_contour.png"
            fig2.savefig(contour_file, dpi=300, bbox_inches="tight")
            plt.close(fig2)
            
            processed_count += 1

        except Exception as e:
            print(f"Error processing trajectory for {key}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Finished generating Loss Landscape Trajectory plots. {processed_count} plots saved to {plots_dir}")


def plot_single_trajectory(trajectory_file, output_dir):
    """Process a single trajectory file"""
    print(f"Loading trajectory from {trajectory_file}...")
    try:
        with open(trajectory_file, "rb") as f:
            trajectory_data = pickle.load(f)
        
        # Extract key information from filename
        path = Path(trajectory_file)
        job_id = path.parent.name
        loss_function = path.parent.parent.name
        program_name = path.parent.parent.parent.name
        
        # Format as expected by plot_loss_landscape_trajectory
        trajectories = {(program_name, loss_function, job_id): trajectory_data}
        
        plot_loss_landscape_trajectory(trajectories, output_dir)
        return True
    except Exception as e:
        print(f"Error processing {trajectory_file}: {e}")
        import traceback
        traceback.print_exc()
        return False


def plot_multiple_trajectories(trajectory_files, output_dir, output_filename="comparison_plot.png"):
    """Plot multiple trajectories on the same 3D plot for comparison.
    
    Args:
        trajectory_files: List of paths to trajectory files
        output_dir: Directory to save the output plot
        output_filename: Name of the output plot file
    """
    print(f"Plotting comparison of {len(trajectory_files)} trajectories...")
    
    # Create output directory
    plots_dir = Path(output_dir) / "plots" / "comparisons"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Load all trajectories
    trajectories_data = []
    trajectory_labels = []
    trajectory_info = []  # Store program_name, loss_function, job_id
    
    for trajectory_file in trajectory_files:
        try:
            path = Path(trajectory_file)
            job_id = path.parent.name
            loss_function = path.parent.parent.name
            program_name = path.parent.parent.parent.name
            label = f"{program_name}/{loss_function}/{job_id}"
            
            with open(trajectory_file, "rb") as f:
                trajectory_data = pickle.load(f)
                
                # Basic validation
                if isinstance(trajectory_data, list) and all(isinstance(item, tuple) and len(item) == 3 for item in trajectory_data):
                    trajectories_data.append(trajectory_data)
                    trajectory_labels.append(label)
                    trajectory_info.append((program_name, loss_function, job_id))
                else:
                    print(f"Warning: Invalid trajectory format in {trajectory_file}")
                    
        except Exception as e:
            print(f"Error loading trajectory {trajectory_file}: {e}")
    
    if not trajectories_data:
        print("No valid trajectories found for comparison.")
        return False
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Different colors for each trajectory
    colors = plt.cm.tab10.colors
    
    # Step 1: Extract and flatten parameters from all trajectories for combined PCA
    all_flattened_params = []
    trajectory_param_indices = []  # Keep track of which params belong to which trajectory
    
    for trajectory_idx, trajectory in enumerate(trajectories_data):
        params_list = [item[1] for item in trajectory]
        
        # Check if params are valid
        if not all(isinstance(p, (dict, hk.Params)) for p in params_list):
            print(f"Warning: Skipping {trajectory_labels[trajectory_idx]} - trajectory contains non-parameter objects.")
            continue
        
        # Flatten parameters for this trajectory
        flattened_params_for_traj = []
        for params in params_list:
            leaves = jax.tree_util.tree_leaves(params)
            if not all(isinstance(leaf, (jnp.ndarray, np.ndarray)) for leaf in leaves):
                print(f"Warning: Skipping {trajectory_labels[trajectory_idx]} - non-array leaf found in parameters.")
                flattened_params_for_traj = None
                break
            if leaves:
                flattened_params_for_traj.append(jnp.concatenate([jnp.ravel(leaf) for leaf in leaves]))
            else:
                flattened_params_for_traj.append(jnp.array([]))
        
        if flattened_params_for_traj is None:
            continue
            
        # Ensure all flattened vectors have the same dimension
        if len(set(p.shape[0] for p in flattened_params_for_traj)) > 1:
            print(f"Warning: Skipping {trajectory_labels[trajectory_idx]} - inconsistent parameter vector lengths.")
            continue
            
        # Record index range for this trajectory in the combined list
        start_idx = len(all_flattened_params)
        all_flattened_params.extend(flattened_params_for_traj)
        end_idx = len(all_flattened_params)
        trajectory_param_indices.append((start_idx, end_idx, params_list))
    
    if not all_flattened_params:
        print("No valid parameters found for comparison.")
        return False
        
    # Check if all parameter vectors have the same dimension
    vector_dimensions = set(p.shape[0] for p in all_flattened_params)
    if len(vector_dimensions) > 1:
        print(f"Warning: Cannot compute combined PCA - inconsistent parameter dimensions {vector_dimensions}")
        return False
        
    # Step 2: Compute combined PCA
    print("Computing combined PCA for all trajectories...")
    param_matrix = jnp.stack(all_flattened_params)
    pca = PCA(n_components=2)
    all_projected_params = pca.fit_transform(param_matrix)
    
    # Step 3: Create combined grid for loss landscape
    min_x, max_x = all_projected_params[:, 0].min(), all_projected_params[:, 0].max()
    min_y, max_y = all_projected_params[:, 1].min(), all_projected_params[:, 1].max()
    
    # Extend the grid beyond the trajectory points
    range_x = max_x - min_x
    range_y = max_y - min_y
    grid_range_factor = 0.2
    min_x -= range_x * grid_range_factor
    max_x += range_x * grid_range_factor
    min_y -= range_y * grid_range_factor
    max_y += range_y * grid_range_factor
    
    # Use smaller grid size for multiple trajectories to reduce computation time
    grid_size = 100
    x_grid = np.linspace(min_x, max_x, grid_size)
    y_grid = np.linspace(min_y, max_y, grid_size)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Plot each trajectory using the shared PCA
    for i, (start_idx, end_idx, params_list) in enumerate(trajectory_param_indices):
        try:
            label = trajectory_labels[i]
            color = colors[i % len(colors)]
            program_name, loss_function, job_id = trajectory_info[i]
            
            # Get trajectory-specific data
            trajectory = trajectories_data[i]
            steps = np.array([item[0] for item in trajectory])
            losses = np.array([item[2] for item in trajectory]).astype(float)
            
            # Get PCA-projected params for this trajectory
            trajectory_pc2d = all_projected_params[start_idx:end_idx]
            
            # Step 4: Compute loss landscape for this trajectory
            job_id_clean = job_id.replace("job_", "")
            Z_grid = np.zeros_like(X_grid)
            
            try:
                # Set up model and data for loss computation
                program_name_key = program_name
                if program_name == "most_freq":
                    program_name_key = "most-freq"
                elif program_name == "shuffle_dyck":
                    program_name_key = "shuffle_dyck1"
                
                # Load the buggy model
                model = load_buggy_models(
                    max_length=10,
                    program_name=program_name,
                    job_id=job_id_clean
                )[job_id_clean]
                
                # Load dataset
                data_path = f"{Path(__file__).parent.resolve()}/../../data/"
                val_dataset = load_dataset(data_path, program_name_key, split_name="val")
                
                # Encode the dataset
                X_val, Y_val = encodeAndPadData(
                    val_dataset, model.raspFunction, model.inputs, 10
                )
                
                # Get loss function
                if loss_function not in LOSS_FUNCTIONS:
                    print(f"Warning: Loss function {loss_function} not found for {label}, using default")
                    loss_fn = cross_entropy_loss
                else:
                    loss_fn = LOSS_FUNCTIONS[loss_function]
                
                # 5. Set up forward pass for loss computation
                padToken = model.model.input_encoder.encoding_map["compiler_pad"]
                
                @jax.jit
                def compute_loss_at_params(params):
                    # Reconstruct the model's parameter structure using first params as template
                    reconstructed_params = jax.tree_util.tree_unflatten(
                        jax.tree_util.tree_structure(params_list[0]), 
                        jax.tree_util.tree_leaves(params)
                    )
                    
                    # Define forward pass
                    def forward(x):
                        compiled_model = model.model.get_compiled_model()
                        compiled_model.use_unembed_argmax = False
                        compiled_model.pad_token = padToken
                        return compiled_model(x, use_dropout=False)
                    
                    # Compute loss on a subset of validation data for speed
                    subset_size = min(256, len(X_val))
                    X_subset = X_val[:subset_size]
                    Y_subset = Y_val[:subset_size]
                    
                    return loss_fn.apply(reconstructed_params, X_subset, Y_subset, padToken, forward=forward)
                
                # Compute loss for a subset of grid points (for efficiency)
                sample_rate = 2  # Only compute every nth point
                print(f"Computing loss landscape for {label} (sampling at 1/{sample_rate} points)...")
                for i in range(0, grid_size, sample_rate):
                    for j in range(0, grid_size, sample_rate):
                        # Convert PCA grid point back to parameter space
                        pca_point = np.array([X_grid[i, j], Y_grid[i, j]])
                        param_vector = pca.inverse_transform([pca_point])[0]
                        
                        try:
                            # Check if the param_vector length matches the flattened params
                            param_structure = jax.tree_util.tree_structure(params_list[0])
                            param_leaves = jax.tree_util.tree_leaves(params_list[0])
                            
                            if len(param_vector) != sum(leaf.size for leaf in param_leaves):
                                Z_grid[i, j] = np.nan
                                continue
                            
                            # Reshape into nested parameter structure
                            reconstructed_params = reconstruct_params(param_vector, params_list[0])
                            
                            # Compute loss at this parameter point
                            loss_value = compute_loss_at_params(reconstructed_params)
                            Z_grid[i, j] = loss_value
                            
                            # Copy value to surrounding points (simple interpolation)
                            for di in range(sample_rate):
                                for dj in range(sample_rate):
                                    if i+di < grid_size and j+dj < grid_size:
                                        Z_grid[i+di, j+dj] = loss_value
                        except Exception as e:
                            Z_grid[i, j] = np.nan
                
                # Fill remaining NaN values with interpolation
                mask = np.isnan(Z_grid)
                Z_grid[mask] = np.nanmean(Z_grid)
                
                # Apply filtering/smoothing for visualization
                from scipy.ndimage import gaussian_filter
                Z_grid = gaussian_filter(Z_grid, sigma=1.0)
                
                # Clip extreme values for better visualization
                p95 = np.nanpercentile(Z_grid, 95)
                Z_grid = np.clip(Z_grid, None, p95 * 1.5)

                # Mask Z_grid values above ceil of initial loss
                initial_loss = trajectory[0][2]
                loss_threshold = np.ceil(initial_loss)
                Z_grid[Z_grid > loss_threshold] = np.nan
                
                # Ensure NaNs are present for transparency
                cmap = plt.cm.viridis.copy()
                cmap.set_bad(color=(0, 0, 0, 0))  # Transparent for masked
                
                # Plot the loss landscape as a surface with partial transparency
                alpha = 0.3  # Use lower alpha for multiple surfaces
                surf = ax.plot_surface(X_grid, Y_grid, Z_grid, 
                                     cmap=cmap, alpha=alpha, linewidth=0,
                                     antialiased=True, zorder=1)
                
            except Exception as e:
                print(f"Error computing loss landscape for {label}: {e}")
                import traceback
                traceback.print_exc()
            
            # Plot the trajectory path
            ax.plot(trajectory_pc2d[:, 0], trajectory_pc2d[:, 1], losses,
                    marker='o', color=color, linestyle='-', linewidth=2, markersize=5, label=label, zorder=2)
                    
            # Plot start and end points
            ax.scatter(trajectory_pc2d[0, 0], trajectory_pc2d[0, 1], losses[0],
                       c=color, s=100, marker='X', depthshade=False, zorder=3)
            ax.scatter(trajectory_pc2d[-1, 0], trajectory_pc2d[-1, 1], losses[-1],
                       c=color, s=100, marker='*', depthshade=False, zorder=3)
                       
        except Exception as e:
            print(f"Error processing trajectory for {trajectory_labels[i]}: {e}")
            import traceback
            traceback.print_exc()
    
    # Add labels and title
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Loss')
    ax.set_title('Comparison of Multiple Trajectories with Loss Landscape')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))
    
    # Save plot
    output_file = plots_dir / output_filename
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Comparison plot saved to {output_file}")
    return True


@click.group()
def cli():
    """Generate trajectory plots for optimization paths."""
    pass


@cli.command()
@click.argument('trajectory_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def plot_file(trajectory_file, output_dir):
    """Plot a single trajectory file."""
    if plot_single_trajectory(trajectory_file, output_dir):
        print(f"Successfully plotted trajectory from {trajectory_file}")
    else:
        print(f"Failed to plot trajectory from {trajectory_file}")


@cli.command()
@click.option(
    "--data-dir",
    required=True,
    help="Directory containing trajectory files (will search recursively)",
    type=click.Path(exists=True)
)
@click.option(
    "--output-dir",
    required=True,
    help="Directory to save plot outputs",
    type=click.Path()
)
def plot_directory(data_dir, output_dir):
    """Process and plot all trajectory files in a directory."""
    print(f"Scanning for trajectory files in {data_dir}...")
    
    # Find all trajectory.pkl files
    try:
        trajectory_files = list(Path(data_dir).rglob("trajectory.pkl"))
    except Exception as e:
        print(f"Error scanning directory {data_dir}: {e}")
        return

    if not trajectory_files:
        print(f"No trajectory.pkl files found in {data_dir} or its subdirectories.")
        return

    print(f"Found {len(trajectory_files)} trajectory.pkl files. Applying filters and processing one by one...")
    
    plotted_count = 0
    processed_file_count = 0

    for trajectory_file_path_obj in trajectory_files:
        processed_file_count += 1
        trajectory_file_path_str = str(trajectory_file_path_obj)
        
        try:
            # Extract info from path for filtering
            # Path structure: .../program_name/loss_function_name/job_id/trajectory.pkl
            parts = trajectory_file_path_obj.parts
            if len(parts) < 4: # Ensure path is deep enough
                print(f"Warning: Path {trajectory_file_path_str} is not in the expected format. Skipping.")
                continue
                
            print(f"Processing and plotting trajectory: {trajectory_file_path_str}")
            # plot_single_trajectory loads the file, creates the dict, and calls plot_loss_landscape_trajectory
            if plot_single_trajectory(trajectory_file_path_str, output_dir):
                plotted_count += 1
            # plot_single_trajectory has its own success/failure print messages.

        except IndexError:
            print(f"Warning: Could not parse path structure for {trajectory_file_path_str}. Skipping.")
        except Exception as e:
            # Catch any other unexpected errors for this specific file to allow the loop to continue
            print(f"Warning: An unexpected error occurred while processing file {trajectory_file_path_str}: {e}. Skipping.")
            # import traceback # Already imported at module level or in other functions
            # traceback.print_exc() # Enable for more detailed debugging if needed

    if plotted_count > 0:
        print(f"\nFinished. Successfully plotted {plotted_count} trajectory/trajectories.")


@cli.command()
@click.argument('trajectory_files', nargs=-1, type=click.Path(exists=True))
@click.option(
    "--output-dir",
    required=True,
    help="Directory to save the comparison plot",
    type=click.Path()
)
@click.option(
    "--output-filename",
    default="comparison_plot.png",
    help="Filename for the comparison plot",
    type=str
)
def compare(trajectory_files, output_dir, output_filename):
    """Plot multiple trajectories on the same graph for comparison.
    
    TRAJECTORY_FILES: Paths to multiple trajectory.pkl files to compare
    """
    if not trajectory_files or len(trajectory_files) < 2:
        print("Error: At least two trajectory files are required for comparison.")
        return
        
    if plot_multiple_trajectories(trajectory_files, output_dir, output_filename):
        print(f"Successfully plotted comparison of {len(trajectory_files)} trajectories")
    else:
        print("Failed to create comparison plot")


if __name__ == "__main__":
    cli()