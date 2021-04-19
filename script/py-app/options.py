import configparser
import sys
# sys.path.append('../option')
# print(sys.path)

from option.dgc import config_dgc


class config_bcfl:
    def __init__(self, config):
        self.config = dict(config._sections["bcfl"])

    def get_sender(self):
        return self.config["sender"]

    def get_db(self):
        return self.config["db"]

    def get_app_port(self) -> int:
        return int(self.config["app_port"])

    def get_save_path(self):
        return self.config["save_path"]
    
    def get_scale_nodes(self) -> int:
        return int(self.config["scale_nodes"])


class config_trainer:
    def __init__(self, config):
        self.config = dict(config._sections["trainer"])

    def get_device(self):
        return self.config["device"]

    def get_dataset(self):
        return self.config["dataset"]

    def get_dataset_path(self):
        return self.config["dataset_path"]

    def get_local_bs(self) -> int:
        return int(self.config["local_bs"])

    def get_local_ep(self) -> int:
        return int(self.config["local_ep"])

    def get_frac(self) -> float:
        return float(self.config["frac"])

    def get_lr(self) -> float:
        return float(self.config["lr"])

    def get_optimizer(self):
        return self.config["optimizer"]

    def get_lossfun(self):
        return self.config["lossfun"]

    def get_max_iteration(self) -> int:
        return int(self.config["max_iteration"])
    
    # warmup
    def get_start_lr(self) -> float:
        return float(self.config["start_lr"])

    def get_max_lr(self) -> float:
        return float(self.config["max_lr"])
    
    def get_min_lr(self) -> float:
        return float(self.config["min_lr"])
    
    def get_base_step(self) -> int:
        return int(self.config["base_step"])
    
    def get_end_step(self) -> int:
        return int(self.config["end_step"])

class config_eval:
    def __init__(self, config):
        self.config = dict(config._sections["eval"])

    def get_ipfsaddr(self):
        return self.config["ipfsaddr"]

    def get_output(self):
        return self.config["output"]

    def get_title(self):
        return self.config["title"]

class config_agg:
    def __init__(self, config):
        self.config = dict(config._sections["aggregator"])

    def get_threshold(self) -> int:
        return int(self.config["threshold"])


class Configer():
    def __init__(self, configfile):
        self.config = configparser.ConfigParser()
        self.config.read(configfile)

        self.bcfl = config_bcfl(self.config)
        self.trainer = config_trainer(self.config)
        self.eval = config_eval(self.config)
        self.agg = config_agg(self.config)

        self.dgc = config_dgc(self.config)
