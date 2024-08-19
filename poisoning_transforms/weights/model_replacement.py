import torch

def weight_replacement_poisoning(weights: torch.Tensor, global_weights: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Apply Weight Replacement Schema using the given formula for poisoning in Federated Learning.

    Parameters:
        weights (torch.Tensor): The local model's weight vector (X).
        global_weights (torch.Tensor): The global model's weight vector (G^t).
        gamma (float): The scaling factor (Γ).

    Returns:
        torch.Tensor: The poisoned weight vector.
    """
    poisoned_weights = gamma * (weights - global_weights) + global_weights
    return poisoned_weights
