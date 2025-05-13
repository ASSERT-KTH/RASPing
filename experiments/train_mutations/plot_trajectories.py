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
from scipy.interpolate import RegularGridInterpolator

# Add parent directory to path for imports
module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.loss import (
    cross_entropy_loss,
    cross_entropy_loss_smoothed_accuracy,
    cross_entropy_loss_with_perfect_sequence,
)
from experiments.mutation.load_mutations import load_buggy_models
from src.functions import load_dataset, encodeAndPadData

# Dictionary of loss functions matching the one in train_mutations.py
LOSS_FUNCTIONS = {
    "cross_entropy_loss": cross_entropy_loss,
    "cross_entropy_loss_smoothed_accuracy": cross_entropy_loss_smoothed_accuracy,
    "cross_entropy_loss_with_perfect_sequence": cross_entropy_loss_with_perfect_sequence,
}


def compute_loss_landscape(trajectory, program_name, loss_function_name, job_id, pca, grid_size=200, grid_range_factor=0.2, pca_grid_limits=None):
    """Compute the loss landscape around the trajectory in PCA space
    
    Args:
        trajectory: List of (step, params, loss) tuples
        program_name: Name of the program model
        loss_function_name: Name of the loss function
        job_id: Job ID for the specific model
        pca: Fitted PCA object to use for projection/inverse
        grid_size: Number of points in each dimension of the grid
        grid_range_factor: How much to extend beyond the min/max of trajectory points
        pca_grid_limits: (min_x, max_x, min_y, max_y) for the grid in PCA space (optional)
        
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
    
    # 2. Project using provided PCA
    projected_params = pca.transform(param_matrix)
    
    # 3. Create a grid in PCA space
    if pca_grid_limits is not None:
        min_x, max_x, min_y, max_y = pca_grid_limits
    else:
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
    loss_threshold = 5 * np.ceil(initial_loss)
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
            pca.fit(param_matrix)
            projected_params = pca.transform(param_matrix)
            trajectory_pc2d = projected_params

            # 4. Create two separate figures - one for surface, one for contour
            # Surface plot
            fig1 = plt.figure(figsize=(12, 8))
            ax1 = fig1.add_subplot(111, projection='3d')
            
            # Contour plot
            fig2 = plt.figure(figsize=(12, 8))
            ax2 = fig2.add_subplot(111)
            
            # Compute and plot loss landscape grid
            landscape_result = compute_loss_landscape(trajectory, program_name, loss_function, job_id, pca=pca)
            
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
                                     cmap=cmap, alpha=0.7, linewidth=0,
                                     antialiased=True, zorder=1)
                
                # Plot contour
                contour = ax2.contour(X_grid, Y_grid, Z_grid_plot, 
                                    levels=20, colors='k', alpha=0.7)
                contourf = ax2.contourf(X_grid, Y_grid, Z_grid_plot, 
                                      levels=20, cmap=cmap, alpha=1.0)
                fig2.colorbar(contourf, ax=ax2, shrink=0.6, aspect=20, label='Correctness Loss')

                # --- Interpolate Z values for trajectory points so the trajectory sits on the surface ---
                interpolator = RegularGridInterpolator((Y_grid[:, 0], X_grid[0, :]), Z_grid_plot, bounds_error=False, fill_value=np.nan)
                trajectory_z_on_surface = interpolator(trajectory_pc2d[:, [1, 0]])  # order is (y, x)

            # 5. Select points for showing examples (start, 25%, 75%, end) based on trajectory distance
            # Compute cumulative distances along the trajectory in PCA space
            diffs = np.diff(trajectory_pc2d, axis=0)
            step_distances = np.linalg.norm(diffs, axis=1)
            cumulative_distances = np.concatenate([[0], np.cumsum(step_distances)])
            total_distance = cumulative_distances[-1]
            # Find indices closest to 25% and 75% of the total distance
            percentiles = [0.0, 0.25, 0.75, 1.0]
            example_indices = []
            for p in percentiles:
                target_dist = p * total_distance
                idx = np.argmin(np.abs(cumulative_distances - target_dist))
                example_indices.append(idx)
            example_markers = ['X', 'o', 'o', 'o']
            example_colors = ['red', '#ff9900', '#99cc00', 'green']  # Red to green
            example_labels = ["Buggy Program", None, None, "Repaired Program"]
            
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
                
                # Only include the first two examples of length 7 (before removing BOS token)
                filtered_examples = []
                for ex in val_dataset:
                    # ex is (input_seq, true_output)
                    input_seq, true_output = ex
                    if hasattr(input_seq, '__len__') and len(input_seq) == 7:
                        filtered_examples.append(ex)
                    if len(filtered_examples) == 2:
                        break
                example_inputs = filtered_examples
                
                # Plot trajectory and examples on both plots
                for plot_idx, ax in enumerate([ax1, ax2]):
                    # Plot full trajectory
                    if plot_idx == 0:  # 3D surface plot
                        ax.plot(trajectory_pc2d[:, 0], trajectory_pc2d[:, 1], trajectory_z_on_surface, 
                               color='r', linestyle='-', linewidth=2, label='Repair Trajectory', zorder=2)
                    else:  # 2D contour plot
                        ax.plot(trajectory_pc2d[:, 0], trajectory_pc2d[:, 1], 
                               color='r', linestyle='-', linewidth=2, label='Repair Trajectory', zorder=2)
                    
                    # Plot example points (markers) in both plots, but add textboxes only in contour plot
                    for idx, (i, marker, color, label) in enumerate(zip(example_indices, example_markers, example_colors, example_labels)):
                        # Plot point (marker)
                        if plot_idx == 0:  # 3D surface plot
                            ax.scatter(trajectory_pc2d[i, 0], trajectory_pc2d[i, 1], trajectory_z_on_surface[i],
                                       s=50, marker=marker, color=color, label=label, depthshade=False, zorder=3)
                        else:  # 2D contour plot
                            ax.scatter(trajectory_pc2d[i, 0], trajectory_pc2d[i, 1],
                                       s=50, marker=marker, color=color, label=label, zorder=3)
                            # Add example textbox (only on contour plot)
                            model.model.params = params_list[i]
                            example_text = ""
                            for input_seq, true_output in example_inputs:
                                output_seq = model.apply(input_seq)
                                clean_input_tokens = [str(token) for token in input_seq if str(token).upper() != "BOS"]
                                original_clean_output_tokens = [str(token) for token in output_seq if str(token).upper() != "BOS"]
                                original_clean_true_output_tokens = [str(token) for token in true_output if str(token).upper() != "BOS"]
                                display_input_seq = " ".join(clean_input_tokens)
                                # Mark the *entire* output as correct/incorrect at the end, based on full sequence match
                                output_parts = [str(token) for token in original_clean_output_tokens]
                                display_output_seq = " ".join(output_parts)
                                if not display_output_seq:
                                    display_output_seq = "(empty output)"
                                # Determine if the full output matches the true output
                                if original_clean_output_tokens == original_clean_true_output_tokens:
                                    mark = "(✓)"
                                else:
                                    mark = "(✗)"
                                example_text += f"{program_name}({display_input_seq}) = {display_output_seq} {mark}\n"
                            example_text = example_text.rstrip("\n")
                            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                            # Improved annotation placement to avoid overlap using smart repulsion

                            # Get plot bounds
                            x_min, x_max = X_grid.min(), X_grid.max()
                            y_min, y_max = Y_grid.min(), Y_grid.max()
                            def clamp(val, minval, maxval):
                                return max(min(val, maxval), minval)

                            # Gather all annotation anchor points for this plot
                            anchor_points = np.array([trajectory_pc2d[j, :2] for j in example_indices])

                            # Define the four corners of the plot
                            corners = np.array([
                                [x_min, y_min],  # bottom-left
                                [x_max, y_min],  # bottom-right
                                [x_min, y_max],  # top-left
                                [x_max, y_max],  # top-right
                            ])

                            # For the first time through, assign each example to the closest corner
                            # Do this only once per plot, not per example
                            if idx == 0:
                                # Compute distance matrix: shape (num_examples, 4)
                                dists = np.linalg.norm(anchor_points[:, None, :] - corners[None, :, :], axis=2)
                                # Find the optimal assignment (minimize total distance)
                                # Use a simple greedy assignment since there are only 4 examples/corners
                                assigned_corners = [-1] * len(anchor_points)
                                assigned_examples = [-1] * 4
                                dists_copy = dists.copy()
                                for _ in range(4):
                                    ex_idx, corner_idx = np.unravel_index(np.argmin(dists_copy), dists_copy.shape)
                                    assigned_corners[ex_idx] = corner_idx
                                    assigned_examples[corner_idx] = ex_idx
                                    dists_copy[ex_idx, :] = np.inf
                                    dists_copy[:, corner_idx] = np.inf
                                # Store the assignment for this plot
                                plot_corner_assignment = assigned_corners
                            # Use the stored assignment
                            corner_idx = plot_corner_assignment[idx]
                            corner = corners[corner_idx]

                            # Offset the annotation a bit away from the corner, toward the example point
                            # Compute a vector from the corner to the example point, normalize, and scale
                            ex_point = anchor_points[idx]
                            vec = ex_point - corner
                            if np.linalg.norm(vec) > 0:
                                vec = vec / np.linalg.norm(vec)
                            offset_magnitude = 0.12 * np.array([x_max - x_min, y_max - y_min])
                            offset_vec = vec * offset_magnitude

                            # Compute annotation position and clamp to bounds
                            x_anno = clamp(corner[0] + offset_vec[0], x_min, x_max)
                            y_anno = clamp(corner[1] + offset_vec[1], y_min, y_max)

                            # Choose alignment based on which corner
                            if corner_idx == 0:  # bottom-left
                                ha, va = 'left', 'bottom'
                            elif corner_idx == 1:  # bottom-right
                                ha, va = 'right', 'bottom'
                            elif corner_idx == 2:  # top-left
                                ha, va = 'left', 'top'
                            elif corner_idx == 3:  # top-right
                                ha, va = 'right', 'top'
                            else:
                                ha, va = 'center', 'center'

                            ax.annotate(
                                example_text,
                                xy=(ex_point[0], ex_point[1]),
                                xytext=(x_anno, y_anno),
                                bbox=props,
                                ha=ha,
                                va=va,
                                fontsize=10,
                                arrowprops=dict(arrowstyle="-")
                            )
                
                    # Add labels and title
                    if plot_idx == 0:  # 3D surface plot
                        ax1.set(
                            xticklabels=[],
                            yticklabels=[],
                        )
                    else:  # 2D contour plot
                        ax2.legend(loc='best', fontsize=8)
                        # Remove numbers from axes
                        ax2.set_xticks([])
                        ax2.set_yticks([])
                        ax2.set_xticklabels([])
                        ax2.set_yticklabels([])
                
            except Exception as e:
                print(f"Error generating examples for {key}: {e}")
                import traceback
                traceback.print_exc()
            
            # 7. Save plots
            plot_subdir = plots_dir / program_name / loss_function
            plot_subdir.mkdir(exist_ok=True, parents=True)
            
            # Save surface plot
            surface_file = plot_subdir / f"{job_id}_loss_landscape_surface.pdf"
            fig1.savefig(surface_file, dpi=300, bbox_inches="tight")
            plt.close(fig1)
            
            # Save contour plot with examples
            contour_file = plot_subdir / f"{job_id}_loss_landscape_contour.pdf"
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


def plot_multiple_trajectories(trajectory_files, output_dir, output_filename):
    """Plot multiple trajectories on the same 3D and 2D plots for comparison.
    Args:
        trajectory_files: List of paths to trajectory.pkl files
        output_dir: Directory to save the output plot
        output_filename: Filename for the output plot
    Returns:
        True if successful, False otherwise
    """
    import pickle
    from pathlib import Path
    import numpy as np
    import jax
    import jax.numpy as jnp
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from scipy.interpolate import RegularGridInterpolator

    # 1. Load all trajectories and extract parameter structures
    loaded_trajectories = []
    param_structures = []
    param_shapes = []
    meta_infos = []  # (program_name, loss_function, job_id)
    for trajectory_file in trajectory_files:
        try:
            with open(trajectory_file, "rb") as f:
                trajectory_data = pickle.load(f)
            # Extract meta info from path
            path = Path(trajectory_file)
            job_id = path.parent.name
            loss_function = path.parent.parent.name
            program_name = path.parent.parent.parent.name
            meta_infos.append((program_name, loss_function, job_id))
            loaded_trajectories.append(trajectory_data)
            # Get parameter structure and shapes from first step
            params = trajectory_data[0][1]
            struct = jax.tree_util.tree_structure(params)
            shapes = [np.shape(leaf) for leaf in jax.tree_util.tree_leaves(params)]
            param_structures.append(struct)
            param_shapes.append(shapes)
        except Exception as e:
            print(f"Error loading {trajectory_file}: {e}")
            return False

    # 2. Check all parameter structures and shapes are the same
    first_struct = param_structures[0]
    first_shapes = param_shapes[0]
    for i, (struct, shapes) in enumerate(zip(param_structures, param_shapes)):
        if struct != first_struct or shapes != first_shapes:
            print(f"Error: Trajectory {trajectory_files[i]} has a different parameter structure or shapes.")
            print(f"All trajectories must have the same model architecture and parameter shapes.")
            return False

    # 3. Flatten all parameters for all trajectories for shared PCA (match single-trajectory logic)
    all_flattened_params = []
    trajectory_flattened = []  # List of lists, one per trajectory
    trajectory_steps = []
    trajectory_losses = []
    valid_indices = []
    import haiku as hk
    for idx, trajectory in enumerate(loaded_trajectories):
        params_list = [item[1] for item in trajectory]
        steps = np.array([item[0] for item in trajectory])
        losses = np.array([item[2] for item in trajectory]).astype(float)
        # Check all params are dict or hk.Params
        if not all(isinstance(p, (dict, hk.Params)) for p in params_list):
            print(f"Warning: Skipping trajectory {meta_infos[idx]} - contains non-parameter objects.")
            continue
        flattened_params = []
        for i, params in enumerate(params_list):
            leaves = jax.tree_util.tree_leaves(params)
            if not all(isinstance(leaf, (jnp.ndarray, np.ndarray)) for leaf in leaves):
                print(f"Warning: Skipping trajectory {meta_infos[idx]} - non-array leaf found in parameters at step {steps[len(flattened_params)]}.")
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
            print(f"Warning: Skipping trajectory {meta_infos[idx]} - inconsistent parameter vector lengths.")
            continue
        trajectory_flattened.append(flattened_params)
        trajectory_steps.append(steps)
        trajectory_losses.append(losses)
        all_flattened_params.extend(flattened_params)
        valid_indices.append(idx)
    if not trajectory_flattened:
        print("Error: No valid trajectories to plot after flattening and validation.")
        return False
    all_flattened_params = jnp.stack(all_flattened_params)
    # Only keep meta_infos for valid trajectories
    meta_infos = [meta_infos[i] for i in valid_indices]

    # 4. Fit shared PCA on all flattened params
    pca = PCA(n_components=2)
    pca.fit(all_flattened_params)

    # 5. Project each trajectory into PCA space
    projected_trajectories = []
    for flattened in trajectory_flattened:
        param_matrix = jnp.stack(flattened)
        projected = pca.transform(param_matrix)
        projected_trajectories.append(projected)

    # Compute global min/max for all projected trajectories
    all_proj = np.concatenate(projected_trajectories, axis=0)
    min_x, max_x = all_proj[:, 0].min(), all_proj[:, 0].max()
    min_y, max_y = all_proj[:, 1].min(), all_proj[:, 1].max()
    # Add margin as in single-trajectory (use grid_range_factor)
    range_x = max_x - min_x
    range_y = max_y - min_y
    min_x -= range_x * 0.2
    max_x += range_x * 0.2
    min_y -= range_y * 0.2
    max_y += range_y * 0.2
    pca_grid_limits = (min_x, max_x, min_y, max_y)

    # 6. Compute loss landscape using the first valid trajectory
    program_name, loss_function, job_id = meta_infos[0]
    landscape_result = compute_loss_landscape(
        loaded_trajectories[valid_indices[0]], program_name, loss_function, job_id, pca=pca, pca_grid_limits=pca_grid_limits
    )
    if landscape_result is None:
        print("Error: Could not compute loss landscape.")
        return False
    X_grid, Y_grid, Z_grid, _, _ = landscape_result
    Z_grid_plot = np.copy(Z_grid)
    p95 = np.nanpercentile(Z_grid_plot, 95)
    Z_grid_plot = np.clip(Z_grid_plot, None, p95 * 1.5)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color=(0, 0, 0, 0))

    # 7. Plot all trajectories on the same 3D surface and 2D contour plots (unchanged)
    fig1 = plt.figure(figsize=(12, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    fig2 = plt.figure(figsize=(12, 8))
    ax2 = fig2.add_subplot(111)

    # Plot surface and contour
    surf = ax1.plot_surface(X_grid, Y_grid, Z_grid_plot, cmap=cmap, alpha=0.7, linewidth=0, antialiased=True, zorder=1)
    contour = ax2.contour(X_grid, Y_grid, Z_grid_plot, levels=20, colors='k', alpha=0.7)
    contourf = ax2.contourf(X_grid, Y_grid, Z_grid_plot, levels=20, cmap=cmap, alpha=1.0)

    # Interpolator for Z values
    from scipy.interpolate import RegularGridInterpolator
    interpolator = RegularGridInterpolator((Y_grid[:, 0], X_grid[0, :]), Z_grid_plot, bounds_error=False, fill_value=np.nan)

    # Colors and markers for each trajectory
    import itertools
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    marker_cycle = itertools.cycle(['o', 's', 'D', '^', 'v', 'P', 'X', '*'])

    for idx, (projected, steps, losses, meta) in enumerate(zip(projected_trajectories, trajectory_steps, trajectory_losses, meta_infos)):
        color = next(color_cycle)
        marker = next(marker_cycle)
        label = f"{meta[0]} | {meta[1]} | {meta[2]}"
        # Interpolate Z for 3D plot
        trajectory_z_on_surface = interpolator(projected[:, [1, 0]])
        # Plot trajectory
        ax1.plot(projected[:, 0], projected[:, 1], trajectory_z_on_surface, color=color, linestyle='-', linewidth=2, label=label, zorder=2)
        ax2.plot(projected[:, 0], projected[:, 1], color=color, linestyle='-', linewidth=2, label=label, zorder=2)
        # Mark start/end
        ax1.scatter(projected[0, 0], projected[0, 1], trajectory_z_on_surface[0], s=80, marker=marker, color=color, label=f"Start {label}", zorder=3)
        ax1.scatter(projected[-1, 0], projected[-1, 1], trajectory_z_on_surface[-1], s=120, marker=marker, color=color, edgecolor='k', label=f"End {label}", zorder=3)
        ax2.scatter(projected[0, 0], projected[0, 1], s=80, marker=marker, color=color, label=f"Start {label}", zorder=3)
        ax2.scatter(projected[-1, 0], projected[-1, 1], s=120, marker=marker, color=color, edgecolor='k', label=f"End {label}", zorder=3)

    # Labels and legends
    ax1.legend(loc='best', fontsize=8)
    ax2.legend(loc='best', fontsize=8)

    # Save plots
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    surface_file = output_dir / (output_filename.replace('.pdf', '_surface.pdf'))
    contour_file = output_dir / (output_filename.replace('.pdf', '_contour.pdf'))
    fig1.savefig(surface_file, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    fig2.savefig(contour_file, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved surface plot to {surface_file}")
    print(f"Saved contour plot to {contour_file}")
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
    default="comparison_plot.pdf",
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
