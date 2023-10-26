
from dataclasses import dataclass
from ConfigSpace import Configuration, ConfigurationSpace, Float

@dataclass
class AdamOptimizationParameters:
    INIT_LR : float
    WEIGHT_DECAY : float
    BATCH_SIZE : int
    EPOCHS : int

    def getName(self):
        return ("Adam"+
                "{:.1e}".format(self.INIT_LR) + "_" +
                "{:.1e}".format(self.WEIGHT_DECAY) + "_" +
                str(self.BATCH_SIZE) + "_" +
                str(self.EPOCHS) + "_")

    @staticmethod
    def get_configuration_space() -> ConfigurationSpace:
        return ConfigurationSpace(
            {
                "INIT_LR": Float("INIT_LR", bounds=(1e-05, 1e-02), log=True),
                "WEIGHT_DECAY": Float("WEIGHT_DECAY", bounds=(1e-05, 1e-02), log=True),
            })
