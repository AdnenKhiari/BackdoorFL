import copy
import torch
from poisoning_transforms.data.datapoisoner import DataPoisoner

class LiraGenerator(DataPoisoner):
    def __init__(self,attacker_train_epoch,attacker_train_optimizer_lr,eps,gen_model: torch.nn.Module,client_model: torch.nn.Module,train_loader,label_replacement,client_model_optimizer_lr,alpha=0.5):
        self.eps = eps
        self.attacker_train_epoch = attacker_train_epoch
        self.attack_model = gen_model
        self.assistant_attack_model = copy.deepcopy(gen_model)
        self.client_model = client_model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.alpha =alpha
        self.label_replacement = label_replacement
        self.client_model_optimizer_lr = client_model_optimizer_lr
        self.train_loader = train_loader
        self.attacker_train_optimizer_lr = attacker_train_optimizer_lr

    def clamp(images,min,max):
        return torch.clamp(images,min,max)
        
    def get_poisoned_batch(self,model,images):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = copy.deepcopy(images).to(device)
        noise = model(images) * self.eps
        images = LiraGenerator.clamp(images + noise,0,1)
        return images
    
    def clear_grad(self, model):
        """
            Clear the gradient of model parameters.
        """
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
                
    def transform(self, data):
        self.attack_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attack_model.to(device)
        img = data["image"].to(device)
        with torch.no_grad():
            return {
                "image": self.get_poisoned_batch(self.attack_model,img),
                "label": data["label"]
            }
    
    def train(self):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_model.eval()
        
        # prepare the hypothical model for the attacker to learn better
        tmp_model = copy.deepcopy(self.client_model)
        tmp_model.train()
        tmp_model.to(device)
        tempmodel_optimizer = torch.optim.SGD(tmp_model.parameters(),lr=self.client_model_optimizer_lr)

        self.assistant_attack_model.load_state_dict(self.attack_model.state_dict())
        
        lira_optimizer = torch.optim.SGD(self.assistant_attack_model.parameters(),lr=self.attacker_train_optimizer_lr)
        
        self.assistant_attack_model.to(device)
        self.assistant_attack_model.train()

        for epoch in range(self.attacker_train_epoch):
            total_mixed_loss = 0
            total_attack_loss = 0
            self.attack_model.to(device)
            self.attack_model.eval()

            for data in self.train_loader:
                mixed_loss, attack_loss = self.train_lira_batch(data,device,tmp_model,tempmodel_optimizer,lira_optimizer)
                with torch.no_grad():
                    total_mixed_loss += mixed_loss
                    total_attack_loss += attack_loss
            
            print(f"LIRA Training : Epoch {epoch} Mixed Loss: {total_mixed_loss} Attack Loss: {total_attack_loss}")
            self.attack_model.load_state_dict(self.assistant_attack_model.state_dict())
                       
    def train_lira_batch(self,data,device,tmp_model,tmp_optimizer,lira_optimizer):
        images,labels = data["image"],data["label"]
        images,labels = images.to(device),labels.to(device)
        
        poisoned_images = self.get_poisoned_batch(self.attack_model,images)
        attacked_labels = tmp_model(poisoned_images)
        predicted_labels = tmp_model(images)
        
        poisoned_target = torch.full(attacked_labels.shape,self.label_replacement,dtype=torch.float16).to(device)
        
        tmp_optimizer.zero_grad()
        mixed_loss = self.alpha * self.criterion(attacked_labels,poisoned_target) + (1- self.alpha)*self.criterion(predicted_labels,labels)
        mixed_loss.backward()
        tmp_optimizer.step()
        
        # train target to backdoor the tmpmodel ( who is trained to be injected with the backdoor )
        lira_optimizer.zero_grad()
        poisoned_images = self.get_poisoned_batch(self.assistant_attack_model,images)
        attacked_labels = tmp_model(poisoned_images)
        attack_loss = self.criterion(attacked_labels,poisoned_target)
        attack_loss.backward()
        lira_optimizer.step()
    
        return mixed_loss.detach().cpu().item(),attack_loss.detach().cpu().item()