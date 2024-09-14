import copy
from flwr.common.parameter import ndarrays_to_parameters
import numpy as np
import torch
import torchmetrics

from models.modelbase import ModelBase
from poisoning_transforms.model.masks.grad_masker import GradMask


def train(net: ModelBase, get_trainloader, optimizer, epochs, device: str,pgd,mask_grad: GradMask | None,weights=None):
    """Train the network on the training set using torchmetrics."""
    criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=net.num_classes).to(device)
    net.train()
    net.to(device)
    
    reference_model = copy.deepcopy(net)

    for epoch in range(epochs):
        total_loss = 0.0
        accuracy_metric.reset()
        
        for batch in get_trainloader():
            batch = net.transform_input(batch)
            images, labels = batch["image"], batch["label"]
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
                    
            if mask_grad != None:
                mask_grad.apply(net)
                
            optimizer.step()
                
            # PGD projection step
            if pgd.active:
                print("Using PGD")
                with torch.no_grad():
                    # Flatten both the current model and reference model parameters
                    current_params = torch.nn.utils.parameters_to_vector(net.parameters())
                    reference_params = torch.nn.utils.parameters_to_vector(reference_model.parameters())

                    # Compute the difference between the current and reference model
                    delta = current_params - reference_params

                    # Check norm type and project accordingly
                    if pgd.norm_type == 'inf':
                        delta = delta.clamp(-pgd.eps, pgd.eps)  # L-infinity projection
                    else:
                        norm = delta.norm(p=pgd.norm_type)
                        if norm > pgd.eps:
                            delta = delta * (pgd.eps / norm)  # Lp projection for p != inf

                    # Update model parameters with the projected values
                    updated_params = reference_params + delta

                    # Update net with projected parameters (copy them over after PGD)
                    torch.nn.utils.vector_to_parameters(updated_params, net.parameters())

            # Metrics
            with torch.no_grad():
                total_loss += loss.item()
                accuracy_metric.update(outputs, labels)
        with torch.no_grad():
            avg_loss = total_loss / sum([1 for _ in get_trainloader()])
            epoch_acc = accuracy_metric.compute().item()
            print(f"Epoch {epoch+1}: train loss {avg_loss}, train accuracy {epoch_acc}")


def test(net, get_testloader, device,weights=None):
    """Evaluate the network on the entire test set using torchmetrics."""
    criterion = torch.nn.CrossEntropyLoss(weight=None).to(device)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=net.num_classes).to(device)
    precision_metric = torchmetrics.Precision(task="multiclass",num_classes=net.num_classes).to(device)
    recall_metric = torchmetrics.Recall(task="multiclass",num_classes=net.num_classes).to(device)
    f1_metric = torchmetrics.F1Score(task="multiclass",num_classes=net.num_classes).to(device)
    # precision_metric = torchmetrics.Precision(task="multiclass",num_classes=net.num_classes).to(device)
    net.eval()
    net.to(device)

    total_loss = 0.0
    with torch.no_grad():
        for batch in get_testloader():
            batch = net.transform_input(batch)

            images, labels = batch["image"], batch["label"]
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            accuracy_metric.update(outputs, labels)
            precision_metric.update(outputs, labels)
            recall_metric.update(outputs, labels)
            f1_metric.update(outputs, labels)

    avg_loss = total_loss / sum([1 for _ in get_testloader()])
    accuracy = accuracy_metric.compute().cpu().item()
    precision = precision_metric.compute().cpu().item()
    recall = recall_metric.compute().cpu().item()
    f1= f1_metric.compute().cpu().item()
    
    return avg_loss, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def model_to_parameters(model):
    """Note that the model is already instantiated when passing it here.

    This happens because we call this utility function when instantiating the parent
    object (i.e. the FedAdam strategy in this example).
    """
    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)
    # print("Extracted model parameters!")
    return parameters
