import numpy as np
import torch

def apply_grad_mask(model, mask_grad_list):
    mask_grad_list_copy = iter(mask_grad_list)
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            parms.grad = parms.grad * next(mask_grad_list_copy)
            
def grad_mask_cv(model: torch.nn.Module, dataset_clearn, criterion, ratio=0.5):
        """Generate a gradient mask based on the given dataset"""
        model.train()
        model.zero_grad()

        for participant_id in range(len(dataset_clearn)):

            train_data = dataset_clearn[participant_id]

            for inputs, labels in train_data:
                inputs, labels = inputs.cuda(), labels.cuda()

                output = model(inputs)

                loss = criterion(output, labels)
                loss.backward(retain_graph=True)

        mask_grad_list = []
        grad_list = []
        grad_abs_sum_list = []
        k_layer = 0
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                grad_list.append(parms.grad.abs().view(-1))

                grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())

                k_layer += 1

        grad_list = torch.cat(grad_list).cuda()
        _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
        mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
        mask_flat_all_layer[indices] = 1.0

        count = 0
        percentage_mask_list = []
        k_layer = 0
        grad_abs_percentage_list = []
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                gradients_length = len(parms.grad.abs().view(-1))

                mask_flat = mask_flat_all_layer[count:count + gradients_length ].cuda()
                mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())

                count += gradients_length

                percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

                percentage_mask_list.append(percentage_mask1)

                grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))

                k_layer += 1
      
        model.zero_grad()
        return mask_grad_list
