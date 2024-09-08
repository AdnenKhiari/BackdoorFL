from poisoning_transforms.model.masks.weight_mask import BaseMask


class GradMask(BaseMask):
    def __init__(self):
        super().__init__()

    def fit(self, model,dataset_clean):
        """
        Learns which gradients to mask. Overrides the BaseMask fit method to learn a mask
        specifically for gradients.
        """
        self.mask = []
        
        # Create a mask based on model's parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Example criterion: Mask out gradients based on parameters (just for demonstration)
                mask = (param.abs() > 1e-2).float()  # Binary mask (1 for keep, 0 for mask out)
                self.mask.append(mask)

    def apply(self, model):
        """
        Multiplies the gradient with the corresponding mask from mask_grad_list.
        mask_grad_list should be an iterable of masks (1 to keep, 0 to mask out).
        """
        mask_grad_list_copy = iter(self.mask)

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                device = param.grad.device
                param.grad = param.grad * next(mask_grad_list_copy).to(device)