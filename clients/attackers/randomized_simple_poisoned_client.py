import numpy as np
import wandb
from clients.poisoned_client import PoisonedFlowerClient
from poisoning_transforms.data.datapoisoner import BatchPoisoner, DataPoisoner, DataPoisoningPipeline
from poisoning_transforms.data.patch.RandomizedSimplePatch import RandomizedSimplePatchPoisoner
from poisoning_transforms.model.model_poisoner import ModelPoisoner, ModelPoisoningPipeline
from poisoning_transforms.model.model_replacement import ModelReplacement
from poisoning_transforms.model.norm_scaling import NormScaling

class RandomizedSimplePoisonedClient(PoisonedFlowerClient):
    def __init__(self, node_id,model_cfg,optimizer,batch_poison_num,target_poisoned,batch_size,patch_location_range: tuple, patch_size_range: tuple, patch_val: float,seed,norm_scaling_factor,pgd_conf,grad_filter) -> None:
        data_poisoner = DataPoisoningPipeline([RandomizedSimplePatchPoisoner(patch_location_range,patch_size_range,patch_val,seed)])
        super(RandomizedSimplePoisonedClient,self).__init__(node_id,model_cfg,optimizer,data_poisoner,batch_poison_num,target_poisoned,batch_size,pgd_conf,grad_filter)
        self.norm_scaling_factor = norm_scaling_factor
    def set_parameters(self, parameters):
        # self.model_poisoner = ModelPoisoningPipeline([ModelReplacement(parameters,3)])
        # self.model_poisoner = ModelPoisoningPipeline([])
        self.model_poisoner = ModelPoisoningPipeline([NormScaling(parameters,self.norm_scaling_factor)])
        return super().set_parameters(parameters)
    
    def report_data(self):
        if self.global_run:
            super().report_data()
