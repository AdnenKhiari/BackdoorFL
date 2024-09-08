import torch
import numpy as np
from poisoning_transforms.model.masks.grad_masker import GradMask

class NeuroToxinPoisoner(GradMask):
    def __init__(self,ratio=0.5):
        super().__init__()
        self.ratio = ratio

    def fit(self, model: torch.nn.Module,dataset_clean):
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        model.zero_grad()

        # Determine the device from the model's first parameter
        device = next(model.parameters()).device

        for batch in dataset_clean:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)

            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward(retain_graph=True)  # Accumulate gradients

        # Step 2: Collect gradients and their absolute sums for each parameter
        grad_list = []
        grad_abs_sum_list = []
        for _, param in model.named_parameters():
            if param.requires_grad:
                grad_list.append(param.grad.abs().view(-1))
                grad_abs_sum_list.append(param.grad.abs().view(-1).sum().item())

        # Step 3: Flatten all gradients into a single vector
        grad_list = torch.cat(grad_list)

        # Step 4: Select the top gradients to mask based on the ratio
        _, indices = torch.topk(-1 * grad_list, int(len(grad_list) * self.ratio))
        mask_flat_all_layer = torch.zeros(len(grad_list), device=device)
        mask_flat_all_layer[indices] = 1.0

        # Step 5: Create mask for each layer based on the flattened mask
        self.mask = []
        count = 0
        layer_stats = []  # To store the statistics for each layer

        for layer_idx, (_, param) in enumerate(model.named_parameters()):
            if param.requires_grad:
                gradients_length = len(param.grad.abs().view(-1))
                mask_flat = mask_flat_all_layer[count:count + gradients_length].to(device)
                self.mask.append(mask_flat.reshape(param.grad.size()).to(device))

                # Calculate the number of pruned neurons and their percentage
                num_total_neurons = gradients_length
                num_pruned_neurons = (mask_flat == 0).sum().item()
                prune_percentage = (num_pruned_neurons / num_total_neurons) * 100

                # Save statistics for the layer
                layer_stats.append({
                    'layer': layer_idx,
                    'total_neurons': num_total_neurons,
                    'pruned_neurons': num_pruned_neurons,
                    'prune_percentage': prune_percentage
                })

                count += gradients_length

        # Display the pruning statistics for each layer
        for stats in layer_stats:
            print(f"Layer {stats['layer']} - Total Neurons: {stats['total_neurons']}, "
                  f"Pruned Neurons: {stats['pruned_neurons']} "
                  f"({stats['prune_percentage']:.2f}%)")

        # Clear the gradients after fitting the mask
        model.zero_grad()
