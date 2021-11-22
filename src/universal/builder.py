
from .adversarial import UniversalAdversarialExample
from .data import Data

import torch
import torch_geometric.data

class Builder():
    def __init__(self, search_iterations = 1):
        # Dictionary containing the parameters of the class to build, UniversalAdvesarialExample.
        self.args = dict()

        self.search_iterations              = search_iterations
        self._perturbation_factory          = None
        self._adversarial_loss_factory      = None
        self._similarity_loss_factory       = None
        self._isospectralization_factory    = None
        self._cross_similarity_loss_factory = None
        self._regularizer_factory           = None
        #self._logger_factory                = EmptyLogger

    # Module constants to define arguments for the class to build.
    USETQDM    = "usetqdm"
    
    ADV_COEFF  = "adversarial_coeff"
    REG_COEFF  = "regularization_coeff"
    SIM_COEFF  = "cross_sim_coeff"
    ISO_COEFF  = "isosp_coeff"

    ADV_LOSS_K = "adv_loss_k"
    
    MIN_IT     = "minimization_iterations"
    LEARN_RATE = "learning_rate"
    MODEL_ARGS = "additional_model_args"

    EIGS_ISOSP_NUM     = "eigen_isospectralization_num"
    EIGS_BANDWIDTH_NUM = "eigen_bandwitdth_num" 

    def set_dataset(self, dataset: torch_geometric.data.InMemoryDataset):
        self.args["dataset"] = dataset
        return self
    
    def set_target(self, t: [int]):
        self.args["target"] = t
        return self

    def set_classifier(self, classifier: torch.nn.Module):
        self.args["classifier"] = classifier
        return self  

    def set_targets(self, targets):
        self.args["target"] = targets
        return self

    def set_template_index(self, template: int):
        self.args["mesh_index"] = template
        return self

    def set_perturbation(self, perturbation_factory):
        self._perturbation_factory = perturbation_factory
        return self

    def set_adversarial_loss(self, adv_loss_factory):
        self._adversarial_loss_factory = adv_loss_factory
        return self

    def set_similarity_loss(self, sim_loss_factory):
        self._similarity_loss_factory = sim_loss_factory
        return self

    def set_cross_similarity_loss(self, cross_sim_loss_factory):
        self._cross_similarity_loss_factory = cross_sim_loss_factory
        return self

    def set_regularization_loss(self, regularizer_factory):
        self._regularizer_factory = regularizer_factory
        return self

    def set_isospectralization_loss(self, isospectralization_factory):
        self._isospectralization_factory = isospectralization_factory
        return self

    def build(self, **args:dict) -> UniversalAdversarialExample:
        usetqdm = args.get(self.USETQDM, False)
        self.args[self.MIN_IT] = args.get(self.MIN_IT, 500)
        self.args[self.ADV_COEFF] = args.get(self.ADV_COEFF, 1)
        self.args[self.REG_COEFF] = args.get(self.REG_COEFF, 1)
        self.args[self.SIM_COEFF] = args.get(self.SIM_COEFF, 1)
        self.args[self.ISO_COEFF] = args.get(self.ISO_COEFF, 1)
        self.args[self.LEARN_RATE] = args.get(self.LEARN_RATE, 1e-3)
        self.args[self.MODEL_ARGS] = args.get(self.MODEL_ARGS, dict())
        self.args[self.EIGS_ISOSP_NUM] = args.get(self.EIGS_ISOSP_NUM, 40)
        self.args[self.EIGS_BANDWIDTH_NUM] = args.get(self.EIGS_BANDWIDTH_NUM, 40)
        
        adv_loss_k = args.get(self.ADV_LOSS_K, 0)
        
        # exponential search variables
        start_adv_coeff = self.args[self.ADV_COEFF]
        range_min, range_max = 0, start_adv_coeff
        optimal_example = None 
        exp_search = True # flag used to detected whether it is the 
                          # first exponential search phase, or the binary search phase

        # start search
        for i in range(self.search_iterations):
          midvalue = (range_min+range_max)/2
          c = range_max if exp_search else midvalue 

          print("[{},{}] ; c={}".format(range_min, range_max, c))
          
          # create adversarial example
          self.args[self.ADV_COEFF] = c #NOTE non-consistent state during execution (problematic during concurrent programming)
          adex = UniversalAdversarialExample(**self.args)
          
          # Define perturbation.
          adex.perturbation = self._perturbation_factory(adex.data, self.args["target"], self.args["mesh_index"])
          
          # Define losses.
          adex.adversarial_loss        = self._adversarial_loss_factory(adex.data, adex.perturbation, adex.classifier, adv_loss_k)
          adex.similarity_loss         = self._similarity_loss_factory(adex.data, adex.perturbation)
          adex.cross_similarity_loss   = self._cross_similarity_loss_factory(adex.data, adex.perturbation)
          adex.isospectralization_loss = self._isospectralization_factory(adex.data, adex.perturbation)
          adex.regularization_loss     = self._regularizer_factory(adex.data, adex.perturbation)
          adex.compute(usetqdm=usetqdm)

          # get perturbation
          alpha = adex.perturbation.alpha

          # update best estimation
          if adex.is_successful:
            optimal_example = adex
            print("** Successful sample, out...")
            break

          # update loop variables
          if exp_search and not adex.is_successful:
            range_min = range_max
            range_max = range_max*2
          elif exp_search and adex.is_successful:
            exp_search = False
          else:
            range_max = range_max if not adex.is_successful else midvalue
            range_min = midvalue  if not adex.is_successful else range_min

        # reset the adversarial example to the original state 
        self.args[self.ADV_COEFF] = start_adv_coeff 

        # if unable to find a good c,r pair, return the best found solution
        is_successful = optimal_example is not None
        if not is_successful: optimal_example = adex
        return optimal_example
