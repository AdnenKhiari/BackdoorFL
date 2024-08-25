from clients.poisoned_client import PoisonedFlowerClient
from poisoning_transforms.data.datapoisoner import DataPoisoner, DataPoisoningPipeline
from poisoning_transforms.data.patch.SimplePatch import SimplePatchPoisoner
from poisoning_transforms.model import model_poisoner
from poisoning_transforms.model.model_poisoner import ModelPoisoner, ModelPoisoningPipeline
from poisoning_transforms.model.model_replacement import ModelReplacement

class SimplePoisonedClient(PoisonedFlowerClient):
    def __init__(self,trainloader, vallodaer,model_cfg,optimizer) -> None:
        super(SimplePoisonedClient,self).__init__(trainloader,vallodaer,model_cfg,optimizer)
        self.data_poisoner = DataPoisoningPipeline([SimplePatchPoisoner((20,20),(5,5),1,1)])
    def set_parameters(self, parameters):
        # self.model_poisoner = ModelPoisoningPipeline([ModelReplacement(parameters)])
        self.model_poisoner = ModelPoisoningPipeline([])
        return super().set_parameters(parameters)