import numpy as np


class EarlyStopper:
    """
    EarlyStopper is a utility class to implement early stopping in training neural networks.

    Early stopping is a technique to stop training once the model performance stops improving on a
    held out validation dataset.
    """

    def __init__(self, patience=20, min_delta=0):
        """
        Initializes the EarlyStopper object.

        Parameters:
        - patience (int, default=20): Number of epochs to wait before stopping once the model stops improving.
        - min_delta (float, default=0): Minimum change in validation loss to be considered as improvement.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0  # Counting the epochs without improvement
        self.min_validation_loss = np.inf  # Initializing with a high value for comparison

    def early_stop(self, validation_loss):
        """
        Determines if early stopping condition is met.

        Parameters:
        - validation_loss (float): The loss on the validation dataset for the current epoch.

        Returns:
        - bool: True if early stopping condition is met, False otherwise.
        """

        # If validation loss improves
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        # If validation loss doesn't improve by more than min_delta
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            # If non-improvement count exceeds patience, return True for early stopping
            if self.counter >= self.patience:
                return True
        return False

    def reset(self):
        """
        Resets the counter to 0.

        This can be useful when using the same EarlyStopper object across different training phases or model.
        """

        self.counter = 0