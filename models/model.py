from flwr.common.parameter import ndarrays_to_parameters
import torch
import torchmetrics

from models.modelbase import ModelBase


def train(net: ModelBase, trainloader, optimizer, epochs, device: str):
    """Train the network on the training set using torchmetrics."""
    criterion = torch.nn.CrossEntropyLoss().to(device)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=net.num_classes).to(device)
    net.train()
    net.to(device)
    for epoch in range(epochs):
        total_loss = 0.0
        accuracy_metric.reset()
        
        for batch in trainloader:
            batch = net.transform_input(batch)
            images, labels = batch["image"], batch["label"]
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                total_loss += loss.item()
                accuracy_metric.update(outputs, labels)
        with torch.no_grad():
            avg_loss = total_loss / len(trainloader)
            epoch_acc = accuracy_metric.compute().item()
            print(f"Epoch {epoch+1}: train loss {avg_loss}, accuracy {epoch_acc}")


def test(net, testloader, device):
    """Evaluate the network on the entire test set using torchmetrics."""
    criterion = torch.nn.CrossEntropyLoss().to(device)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=net.num_classes).to(device)
    # precision_metric = torchmetrics.Precision(task="multiclass",num_classes=net.num_classes).to(device)
    net.eval()
    net.to(device)

    total_loss = 0.0
    with torch.no_grad():
        for batch in testloader:
            batch = net.transform_input(batch)

            images, labels = batch["image"], batch["label"]
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            accuracy_metric.update(outputs, labels)
            # precision_metric.update(outputs, labels)

    avg_loss = total_loss / len(testloader)
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