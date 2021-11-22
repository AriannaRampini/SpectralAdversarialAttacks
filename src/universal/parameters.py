
from .builder import Builder

class Parameters:
    def __init__(self):
        self.reset()

    def reset(self):
        self.name: str = None
        
        # Dataset folder, parameters file and portion to use (train|test|custom).
        self.dataset: str = None
        self.dataset_file: str = None
        self.datasubset: str = None
        self.datasubset_list: list = []
        
        # Shape index to use as template.
        self.template_index: int = None
        
        # Targets list (len(targets)==len(dataset), checked via assertion).
        self.targets: list = None

        # Builder parameters
        self.usetqdm = False
        self.minimun_iterations = None
        self.learn_rate = None

        self.adv_coefficient = None
        self.reg_coefficient = None
        self.sim_coefficient = None
        self.iso_coefficient = None    

        self.adv_loss_k = None
    
        self.eigs_isop_num = None
        self.eigs_bandwidth_num = None

        self.model_args = None

    def builder_args(self) -> dict:
        builder_args = dict()
        builder_args[Builder.USETQDM] = self.usetqdm if self.usetqdm else False
        builder_args[Builder.MIN_IT] = self.minimun_iterations if self.minimun_iterations else 500
        builder_args[Builder.ADV_COEFF] = self.adv_coefficient if self.adv_coefficient else 1
        builder_args[Builder.REG_COEFF] = self.reg_coefficient if self.reg_coefficient else 1
        builder_args[Builder.SIM_COEFF] = self.sim_coefficient if self.sim_coefficient else 1
        builder_args[Builder.ISO_COEFF] = self.iso_coefficient if self.iso_coefficient else 1
        builder_args[Builder.LEARN_RATE] = self.learn_rate if self.learn_rate else 1e-3
        builder_args[Builder.MODEL_ARGS] = self.model_args if self.model_args else dict()
        builder_args[Builder.EIGS_ISOSP_NUM] = self.eigs_isop_num if self.eigs_isop_num else 40
        builder_args[Builder.EIGS_BANDWIDTH_NUM] = self.eigs_bandwidth_num if self.eigs_bandwidth_num else 40
        builder_args[Builder.ADV_LOSS_K] = self.adv_loss_k if self.adv_loss_k else 0

        return builder_args

    def check(self):
        if self.name is None:
            raise ValueError("Must specify name of the dataset (e.g. smal).")
        
        if self.dataset is None:
            raise ValueError("Must specify path for the dataset.")

        if self.dataset_file is None:
            raise ValueError("Must specify parameters file path of the correspoding model.")
        
        if self.datasubset is None:
            raise ValueError("Must specify which part of the dataset to use.")
        
        if self.datasubset == "custom" and len(self.datasubset_list) == 0:
            raise ValueError("The subset of shapes to use is empty.")

        if self.template_index is None:
            raise ValueError("Must specify an index to use as template shape.")

        return


from json import JSONEncoder

class ParametersEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__