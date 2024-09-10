import copy
import torch
from poisoning_transforms.data.datapoisoner import DataPoisoner

class LiraGenerator(DataPoisoner):
    def __init__(self,attacker_train_epoch,attacker_train_optimizer_lr,eps,gen_model: torch.nn.Module,client_model: torch.nn.Module,train_loader,label_replacement,client_model_optimizer_lr,alpha=0.5):
        self.eps = eps
        self.attacker_train_epoch = attacker_train_epoch
        self.attack_model = gen_model
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
        self.client_model.to(device)
        
        # prepare the hypothical model for the attacker to learn better
        tmp_model = copy.deepcopy(self.client_model)
        tmp_model.train()
        tmp_model.to(device)
        tempmodel_optimizer = torch.optim.SGD(tmp_model.parameters(),lr=self.client_model_optimizer_lr)
        
        lira_optimizer = torch.optim.SGD(self.attack_model.parameters(),lr=self.attacker_train_optimizer_lr)
        
        self.attack_model.to(device)
        self.attack_model.train()
        k_lira = 2
        for epoch in range(self.attacker_train_epoch):
            total_attack_loss = 0
            total_tempmodel_loss = 0
            self.attack_model.to(device)
            self.attack_model.eval()

            for i in range(k_lira):
                for data in self.train_loader:
                    attack_loss = self.train_lira_batch(data,device,tmp_model,lira_optimizer)
                    with torch.no_grad():
                        total_attack_loss += attack_loss
            total_attack_loss/= k_lira
            
            for data in self.train_loader:
                defense_loss = self.train_temp_model(data,device,tmp_model,tempmodel_optimizer)
                with torch.no_grad():
                    total_tempmodel_loss += defense_loss
            print(f"Epoch {epoch} : Attack Loss : {total_attack_loss} Defense Loss : {total_tempmodel_loss}")
            
    def train_temp_model(self,data,device,tmp_model,tmp_optimizer):
        images,labels = data["image"],data["label"]
        images,labels = images.to(device),labels.to(device)
        
        tmp_optimizer.zero_grad()
        poisoned_images = self.get_poisoned_batch(self.attack_model,images)
        attacked_labels = tmp_model(poisoned_images)
        loss = self.criterion(attacked_labels,labels)
        loss.backward()
        tmp_optimizer.step()
        
    def train_lira_batch(self,data,device,tmp_model,lira_optimizer):
        images,labels = data["image"],data["label"]
        images,labels = images.to(device),labels.to(device)

        # train target to backdoor the tmpmodel ( who is trained to be injected with the backdoor )
        lira_optimizer.zero_grad()
        poisoned_images = self.get_poisoned_batch(self.attack_model,images)
        newly_attacked_labels = tmp_model(poisoned_images)
        standard_attacked_labels = self.client_model(poisoned_images)
        poisoned_labels = torch.tensor([self.label_replacement]*len(newly_attacked_labels)).to(device)
        attack_loss = self.alpha*self.criterion(standard_attacked_labels,poisoned_labels) + (1-self.alpha)*self.criterion(newly_attacked_labels,)
        attack_loss.backward()
        lira_optimizer.step()
    
        return attack_loss.detach().cpu().item()