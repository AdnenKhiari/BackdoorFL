import numpy as np
import wandb
from clients.poisoned_client import PoisonedFlowerClient
from poisoning_transforms.data.datapoisoner import BatchPoisoner, DataPoisoner, DataPoisoningPipeline
from poisoning_transforms.data.generated.generated_models import UNet
from poisoning_transforms.data.generated.lira import LiraGenerator
from poisoning_transforms.data.patch.SimplePatch import SimplePatchPoisoner
from poisoning_transforms.model import model_poisoner
from poisoning_transforms.model.model_poisoner import ModelPoisoner, ModelPoisoningPipeline
from poisoning_transforms.model.model_replacement import ModelReplacement
from poisoning_transforms.model.norm_scaling import NormScaling
from hydra.utils import instantiate

class IbaClient(PoisonedFlowerClient):
    def __init__(self, 
                 node_id,
                 model_cfg,
                 optimizer,
                 optimizer_lr,
                 batch_poison_num,
                 target_poisoned,
                 batch_size,
                 lira_output_size,
                 lira_train_epoch,
                 lira_train_lr,
                 lira_eps,
                 norm_scaling_factor,
                 pgd_conf,
                 mask_conf,
                 poison_between
                 ) -> None:
        
        model = instantiate(model_cfg)
        lira_model = UNet(lira_output_size)
        self.lira = LiraGenerator(
                lira_train_epoch,
                lira_train_lr,
                lira_eps,
                lira_model,
                model,
                None,
                target_poisoned,
                optimizer_lr
            )
        data_poisoner = DataPoisoningPipeline([
                self.lira
            ]
        )
        super(IbaClient,self).__init__(node_id,model_cfg,optimizer,data_poisoner,batch_poison_num,target_poisoned,batch_size,pgd_conf,mask_conf,poison_between)
        self.model = model
        self.norm_scaling_factor = norm_scaling_factor
        
    def with_loaders(self, trainloader, vallodaer):
        self.lira.train_loader = trainloader
        return super().with_loaders(trainloader, vallodaer)
        
    def set_parameters(self, parameters):
        # self.model_poisoner = ModelPoisoningPipeline([ModelReplacement(parameters,3)])
        self.model_poisoner = ModelPoisoningPipeline([NormScaling(parameters,self.norm_scaling_factor)])
        return super().set_parameters(parameters)
    
    def report_data(self):
        if self.global_run:
            super().report_data()
            # wandb.run._set_config_wandb("IbaClient",((20,20),(5,5)))