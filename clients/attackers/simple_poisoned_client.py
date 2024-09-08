import numpy as np
import wandb
from clients.poisoned_client import PoisonedFlowerClient
from poisoning_transforms.data.datapoisoner import BatchPoisoner, DataPoisoner, DataPoisoningPipeline
from poisoning_transforms.data.patch.SimplePatch import SimplePatchPoisoner
from poisoning_transforms.model import model_poisoner
from poisoning_transforms.model.model_poisoner import ModelPoisoner, ModelPoisoningPipeline
from poisoning_transforms.model.model_replacement import ModelReplacement
from poisoning_transforms.model.norm_scaling import NormScaling

class SimplePoisonedClient(PoisonedFlowerClient):
    def __init__(self, node_id,model_cfg,optimizer,batch_poison_num,target_poisoned,batch_size,patch_size,patch_loc,patch_val,norm_scaling_factor,pgd_conf,mask_conf) -> None:
        data_poisoner = DataPoisoningPipeline([SimplePatchPoisoner(patch_size,patch_loc,patch_val)])
        super(SimplePoisonedClient,self).__init__(node_id,model_cfg,optimizer,data_poisoner,batch_poison_num,target_poisoned,batch_size,pgd_conf,mask_conf)
        self.norm_scaling_factor = norm_scaling_factor
    def set_parameters(self, parameters):
        # self.model_poisoner = ModelPoisoningPipeline([ModelReplacement(parameters,3)])
        # self.model_poisoner = ModelPoisoningPipeline([])
        self.model_poisoner = ModelPoisoningPipeline([NormScaling(parameters,self.norm_scaling_factor)])
        return super().set_parameters(parameters)
    
    def report_data(self):
        if self.global_run:
            super().report_data()
