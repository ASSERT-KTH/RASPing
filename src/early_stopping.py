import numpy as np


class EarlyStopping:
    """Early stopping handler to monitor training progress and stop when no improvement is seen.

    Args:
        patience (int): Number of epochs to wait for improvement before stopping
        min_delta (float): Minimum change in monitored value to qualify as an improvement
        mode (str): One of {'min', 'max'}. In min mode, training stops when the quantity monitored
                   stops decreasing; in max mode it will stop when the quantity monitored stops increasing.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = abs(min_delta)  # Make sure min_delta is positive
        self.mode = mode
        self.counter = 0
        self.best_value = np.inf if mode == "min" else -np.inf
        self.best_params = None
        self.best_step = None  # Store the step number when best_params is updated
        self.should_stop = False

        if mode not in ["min", "max"]:
            raise ValueError(f"mode {mode} is unknown, should be 'min' or 'max'")

        self.monitor_op = np.less if mode == "min" else np.greater
        self.min_delta *= 1 if mode == "min" else -1  # Change sign based on mode

    def __call__(self, current_value: float, current_params=None, current_step=None) -> bool:
        """Returns True if training should stop."""
        # For 'min' mode: Checks if current_value + min_delta < best_value
        # For 'max' mode: Checks if current_value - min_delta > best_value
        if self.monitor_op(current_value + self.min_delta, self.best_value):
            self.best_value = current_value
            if current_params is not None:
                self.best_params = current_params
            if current_step is not None:
                self.best_step = current_step
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_value = np.inf if self.mode == "min" else -np.inf
        self.best_params = None
        self.best_step = None
        self.should_stop = False
