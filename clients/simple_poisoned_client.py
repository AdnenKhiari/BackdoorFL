import numpy as np
import wandb
from clients.poisoned_client import PoisonedFlowerClient
from poisoning_transforms.data.datapoisoner import BatchPoisoner, DataPoisoner, DataPoisoningPipeline
from poisoning_transforms.data.patch.SimplePatch import SimplePatchPoisoner
from poisoning_transforms.model import model_poisoner
from poisoning_transforms.model.model_poisoner import ModelPoisoner, ModelPoisoningPipeline
from poisoning_transforms.model.model_replacement import ModelReplacement

class SimplePoisonedClient(PoisonedFlowerClient):
    def __init__(self, node_id,model_cfg,optimizer,batch_poison_num,target_poisoned,batch_size) -> None:
        data_poisoner = DataPoisoningPipeline([SimplePatchPoisoner((20,20),(5,5),1)])
        super(SimplePoisonedClient,self).__init__(node_id,model_cfg,optimizer,data_poisoner,batch_poison_num,target_poisoned,batch_size)
    def set_parameters(self, parameters):
        self.model_poisoner = ModelPoisoningPipeline([])
        # self.model_poisoner = ModelPoisoningPipeline([])
        return super().set_parameters(parameters)
    def report_data(self):
        if self.global_run:
            super().report_data()
            wandb.run._set_config_wandb("SimplePatchPoisoner",((20,20),(5,5)))