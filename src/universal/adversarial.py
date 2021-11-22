
from .base_adversarial import BaseAdversarial
from .data import Data
from .perturbation import UniversalPerturbation
from .loss import UniversalAdversarialLoss, UniversalL2Similarity, UniversarlCrossL2Similarity, UniversalIsospec

import torch
import torch_geometric.data
import tqdm

class UniversalAdversarialExample(BaseAdversarial):
    def __init__(self,
                 dataset: torch_geometric.data.InMemoryDataset,
                 mesh_index: int,
                 classifier: torch.nn.Module,
                 target: [int],
                 adversarial_coeff: float,
                 regularization_coeff: float,
                 cross_sim_coeff: float,
                 isosp_coeff: float,
                 minimization_iterations: int,
                 learning_rate: float,
                 eigen_isospectralization_num: int,
                 eigen_bandwitdth_num: int,
                 additional_model_args: dict):
        super().__init__(classifier = classifier, 
                         target = target,
                         classifier_args = additional_model_args)
        # Data.
        self._data = Data(dataset, isospectralization = eigen_isospectralization_num, lowband = eigen_bandwitdth_num)

        # Template mesh to optimize.
        self._template_shape_index = mesh_index

        # Coefficients.
        self.adversarial_coeff    = torch.tensor([adversarial_coeff], device=self.device, dtype=self.dtype_float)
        self.regularization_coeff = torch.tensor([regularization_coeff], device=self.device, dtype=self.dtype_float)
        self.cross_sim_coeff      = torch.tensor([cross_sim_coeff], device=self.device, dtype=self.dtype_float)
        self.isosp_coeff          = torch.tensor([isosp_coeff], device=self.device, dtype=self.dtype_float)

        # Other parameters.
        self.minimization_iterations = minimization_iterations
        self.learning_rate = learning_rate
        self.model_args = additional_model_args

        # Commponents.
        self.perturbation = None
        
        # Set manually through the Builder class.
        self.adversarial_loss        = lambda : self._zero
        self.similarity_loss         = lambda : self._zero
        self.cross_similarity_loss   = lambda : self._zero
        self.isospectralization_loss = lambda : self._zero
        self.regularization_loss     = lambda : self._zero
        self._zero = torch.tensor(0, dtype=self.dtype_float, device=self.device)
        
        # Rudimentary logger.
        self._logger = dict()

    def pos(self, shape_index):
        return self.data.pos(shape_index)

    @property
    def perturbed_template_pos(self):
        r"""Returns perturbed positions of template shape through alpha."""
        return self.perturbation.perturb_positions()

    def perturbation_alpha(self):
        r"""Returns perturbation through alpha."""
        return self.perturbation.perturbation_alpha()

    def perturbation_alpha_i(self, shape_index):
        r"""Returns perturbation through alpha_i with corresponding base."""
        return self.perturbation.perturbation_alpha_i(shape_index)

    def perturbed_positions_i(self, shape_index):
        r"""Return perturbed positions for a given shape via alpha_i and correspoding base."""
        return self.perturbation.perturb_positions_i(shape_index)

    @property
    def data(self):
        return self._data
    
    @property
    def device(self):
        return self.data.device

    @property
    def dtype_float(self):
        return self.data.dtype_float
    
    @property
    def dtype_int(self):
        return self.data.dtype_int
    
    @property
    def template_shape_index(self):
        return self._template_shape_index

    def faces(self, shape_index):
        return self.data.faces(shape_index) 

    def edges(self, shape_index):
        return self.data.edges(shape_index) 

    def shape_count(self):
        return self.data.shape_count

    def file_name(self, shape_index):
        return self.data.file_name(shape_index)

    @property
    def logger(self):
        return self._logger

    def classify(self, pos):
        return self.classifier(pos.float(), **self._classifier_args)

    def logits(self, shape_index) -> torch.Tensor:
        return self.classify(self.pos(shape_index))

    def perturbed_logits(self, shape_index) -> torch.Tensor:
        perturbed_pos = self.pos(shape_index) + self.perturbation_alpha()
        return self.classify(perturbed_pos)

    @property
    def is_successful(self) -> [bool]:
        output = []
        for shape_index in range(self.data.shape_count):
            prediction = self.logits(shape_index).argmax().item()
            adversarial_prediction = self.perturbed_logits(shape_index).argmax().item()

            output.append(prediction != adversarial_prediction)
        return all(output)

    def reset_logger(self):
        return {"loss": [], "similarity": [], 
                "cross_similarity": [], "adversarial": [], 
                "regularization": [], "isospectralization": []}

    def log(self, loss, sim, cross, adv, reg, iso):
        self.logger["loss"].append(loss)
        self.logger["similarity"].append(sim)
        self.logger["cross_similarity"].append(cross)
        self.logger["adversarial"].append(adv)
        self.logger["regularization"].append(reg)
        self.logger["isospectralization"].append(iso)
        return

    def compute(self, usetqdm: str = None, patience = 3):
        # Reset variables.
        self.perturbation.reset()
        self._logger = self.reset_logger()
        
        # Optimizer, compute gradient w.r.t. the perturbation.
        optimizer = torch.optim.Adam(self.perturbation.variables_to_optimize(), 
                                     lr = self.learning_rate, betas = (0.5, 0.75))

        if usetqdm is None or usetqdm == False:
          iterations = range(self.minimization_iterations)
        elif usetqdm == "standard" or usetqdm == True:
          iterations = tqdm.trange(self.minimization_iterations)
        elif usetqdm == "notebook":
          iterations = tqdm.tqdm_notebook(range(self.minimization_iterations))
        else:
          raise ValueError("Invalid input for 'usetqdm', valid values are: None, 'standard' and 'notebook'.")
        
        flag, counter = False, patience
        self.perturbation.save_last()
        
        for i in iterations:
          # Compute loss.
          optimizer.zero_grad()
          
          # Compute total loss.
          adv_loss = self.adversarial_loss()
          adversarial_loss        = self.adversarial_coeff * adv_loss
          
          iso_loss = self.isospectralization_loss()
          isospectralization_loss = self.isosp_coeff * iso_loss
          
          loss = adversarial_loss + isospectralization_loss

          # Cutoff procedure to improve performance
          is_successful = all(list(map(lambda x: x<=0, self.adversarial_loss.list)))

          if is_successful:
            counter -= 1
            if counter<=0:
                self.perturbation.save_last()
                flag = True
          else: 
            counter = patience

          # backpropagate
          loss.backward()
          optimizer.step()

