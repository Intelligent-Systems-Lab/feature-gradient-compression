import configparser

class config_dgc:
    def __init__(self, config):
        self.config = dict(config._sections["dgc"])

    # def get_dgc(self):
    #     return self.config["dgc"]

    def get_compress_ratio(self) -> float:
        return float(self.config["compress_ratio"])

    def get_fusing_ratio(self) -> float:
        return float(self.config["fusing_ratio"])

    def get_momentum(self) -> float:
        return float(self.config["momentum"])

    def get_momentum_correction(self) -> bool:
        if self.config["momentum_correction"] == "False":
            return False
        elif self.config["momentum_correction"] == "True":
            return True


    # def get_sample_ratio(self) -> float:
    #     return float(self.config["sample_ratio"])

    # def get_strided_sample(self) -> bool:
    #     return bool(self.config["strided_sample"])

    # def get_compress_upper_bound(self) -> float:
    #     return float(self.config["compress_upper_bound"])

    # def get_compress_lower_bound(self) -> float:
    #     return float(self.config["compress_lower_bound"])

    # def get_max_adaptation_iters(self) -> int:
    #     return int(self.config["max_adaptation_iters"])

    # def get_resample(self) -> bool:
    #     return bool(self.config["resample"])
    
    # def get_momentum(self) -> float:
    #     return float(self.config["momentum"])