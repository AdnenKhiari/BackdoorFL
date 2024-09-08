from flwr.common.parameter import ndarrays_to_parameters
import torch
import torchmetrics

from models.modelbase import ModelBase


def train(net: ModelBase, get_trainloader, optimizer, epochs, device: str,pgd,mask_grad):
    """Train the network on the training set using torchmetrics."""
    criterion = torch.nn.CrossEntropyLoss().to(device)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=net.num_classes).to(device)
    net.train()
    net.to(device)
    
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
            
            if pgd.active:
                with torch.no_grad():
                    # Compute and print old norm of gradients before PGD
                    old_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in net.parameters() if p.grad is not None]), pgd.norm_type)
                    print(f"Old gradient norm before PGD: {old_grad_norm}")

                    # Apply PGD by clipping gradient norms
                    torch.nn.utils.clip_grad_norm_(net.parameters(), pgd.eps, pgd.norm_type, error_if_nonfinite=True)

                    # Compute and print new norm of gradients after PGD
                    new_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in net.parameters() if p.grad is not None]), pgd.norm_type)
                    print(f"New gradient norm after PGD: {new_grad_norm}")
                    
            if mask_grad.active:
                mask_grad.apply(net)
            
            optimizer.step()

            # Metrics
            with torch.no_grad():
                total_loss += loss.item()
                accuracy_metric.update(outputs, labels)
        with torch.no_grad():
            avg_loss = total_loss / sum([1 for _ in get_trainloader()])
            epoch_acc = accuracy_metric.compute().item()
            print(f"Epoch {epoch+1}: train loss {avg_loss}, train accuracy {epoch_acc}")


def test(net, get_testloader, device):
    """Evaluate the network on the entire test set using torchmetrics."""
    criterion = torch.nn.CrossEntropyLoss().to(device)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=net.num_classes).to(device)
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
            # precision_metric.update(outputs, labels)

    avg_loss = total_loss / sum([1 for _ in get_testloader()])
    accuracy = accuracy_metric.compute().cpu().item()
    # precision = precision_metric.compute().cpu().item()
    
    return avg_loss, {
        "accuracy": accuracy,
        # "precision": precision
    }


def model_to_parameters(model):
    """Note that the model is already instantiated when passing it here.

    This happens because we call this utility function when instantiating the parent
    object (i.e. the FedAdam strategy in this example).
    """
    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)
    print("Extracted model parameters!")
    return parameters