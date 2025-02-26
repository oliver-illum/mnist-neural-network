import numpy as np

def cross_entropy_loss(predictions, targets, epsilon=1e-8):
    # prevent log(0)
    predictions_clipped = np.clip(predictions, epsilon, 1.0)
    loss_per_example = -np.sum(targets * np.log(predictions_clipped), axis=1)
    return np.mean(loss_per_example)

def cross_entropy_loss_derivative(predictions, targets):
    """
    For softmax combined with cross-entropy, the gradient simplifies to:
    predictions - targets.
    """
    return predictions - targets
