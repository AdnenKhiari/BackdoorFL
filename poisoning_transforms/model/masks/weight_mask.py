import torch

class BaseMask:
    def __init__(self):
        self.mask = None

    def fit(self, model: torch.nn.Module):
        """
        Learns which weights to mask by creating a mask for each parameter that requires gradients.
        For simplicity, let's mask out small weights by creating a binary mask.
        """
        self.mask = []
        
        # Create a mask based on model's parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Example criterion: Mask out weights close to zero
                mask = (param.abs() > 1e-2).float()  # Binary mask (1 for keep, 0 for mask out)
                self.mask.append(mask)

    def apply(self, model):
        """
        Apply the learned mask to the model's parameters.
        """
        mask_iter = iter(self.mask)
        for name, param in model.named_parameters():
            if param.requires_grad:
                mask = next(mask_iter)
                # Apply the mask by element-wise multiplication with the parameter
                param.data = param.data * mask