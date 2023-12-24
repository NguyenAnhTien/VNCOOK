"""
@author : Tien Nguyen
@date   : 2023-Dec-23
"""
import json
import yaml

class Configurer(object):
    def __init__(
            self, config_file: str
        ) -> None:
        self.parse(config_file)
    
    def parse(
            self, 
            config_file
        ) -> None:
        configs = self.read_yaml(config_file)
        for group, value in configs.items():
            if not isinstance(value, dict):
                exec(f"self.{group} = value")
                continue
            for task, behavior in value.items():
                exec(f"self.{task} = behavior")

    def __str__(
            self
        ) -> str:
        return json.dumps({
            "batch_size"        : self.batch_size,
            "lr"                : self.lr,
            "weight_decay"      : self.weight_decay,
            "epochs"            : self.epochs,
            "model_name"        : self.model_name,
            "pretrained"        : self.pretrained,
            "patience"          : self.patience,
        }, indent=4)

    def __repr__(
            self
        ) -> str:
        return self.__str__()

    def read_yaml(
        self,
        file_name: str
    ) -> dict:
        with open(file_name, 'r') as file:
            configs = yaml.safe_load(file)
        return configs
