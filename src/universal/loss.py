
import torch

from .base_data import BaseData
from .perturbation import UniversalPerturbation

class LossFunction(object):
    def __init__(self, data: BaseData, perturbation: UniversalPerturbation):
        self._data = data
        self._perturbation = perturbation

    def __call__(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def shape_count(self) -> int:
        return self._data.shape_count

    def pos(self, shape_index):
        return self._data.pos(shape_index)

    def faces(self, shape_index):
        return self._data.faces(shape_index)

    def y(self, shape_index) -> int:
        """Returns original label classification for a given shape. """
        return self._data.y(shape_index)

    def eigenvectors(self, shape_index):
        return self._data.eigenvectors(shape_index)

    def eigenvalues(self, shape_index):
        return self._data.eigenvalues(shape_index)
    
    @property
    def eigencount(self):
        return self._data.eigen_count

    @property
    def eigencount_isosp(self):
        return self._data.isospectralization

    def area(self, shape_index):
        return self._data.area(shape_index)

    @property
    def alpha(self):
        return self._perturbation.alpha
    
    def alpha_i(self, shape_index):
        return self._perturbation.alpha_i[:, :, shape_index]

    @property
    def epsilon(self):
        return self._perturbation.epsilon

    @property
    def template_eigvecs(self):
        return self._perturbation.template_eigvecs


class ZeroLoss(LossFunction):
    def __init__(self, data: BaseData, perturbation: UniversalPerturbation):
        super().__init__(data, perturbation)
        self._zero = torch.zeros(1, dtype=data.dtype_float, device=data.device)
    
    def __call__(self): 
        return self._zero


class UniversalAdversarialLoss(LossFunction):
    def __init__(self, 
                 data: BaseData, 
                 perturbation: UniversalPerturbation, 
                 classifier: torch.nn.Module, 
                 k: float = 0):
        super().__init__(data, perturbation)
        self.k = torch.tensor([k], device=data.device, dtype=data.dtype_float)
        self._list = []
        self._classifier = classifier

    def target(self, shape_index):
        if self._perturbation.targets is None:
            return None
        else:
            return self._perturbation.targets[shape_index]

    def classify(self, pos):
        return self._classifier(pos.float())

    @property
    def list(self):
        return self._list

    def adversarial_loss(self, perturbed_logits, y: int) -> torch.Tensor:
        Z = perturbed_logits
        values, index = torch.sort(Z, dim=0)
        argmax = index[-1] if index[-1] != y else index[-2] 
        Zy, Zmax = Z[y], Z[argmax]

        return self.k + torch.max(Zy - Zmax, -self.k)

    def adversarial_loss_w_target(self, perturbed_logits, target: int) -> torch.Tensor:
        Z = perturbed_logits
        values, index = torch.sort(Z, dim=0)
        argmax = index[-1] if index[-1] != target else index[-2] # max{Z(i): i != target}
        Ztarget, Zmax = Z[target], Z[argmax]
        
        return self.k + torch.max(Zmax - Ztarget, -self.k)

    def __call__(self) -> torch.Tensor:
        summation = 0
        sum_list = []
        for shape_index in range(self.shape_count):

            target = self.target(shape_index)
            
            eigvecs_i = self.eigenvectors(shape_index)
            perturbation = eigvecs_i.matmul(self.alpha_i(shape_index))
            perturbed_pos = self.pos(shape_index) + perturbation
            
            perturbed_logits = self.classify(perturbed_pos)
            perturbed_logits = perturbed_logits.double()

            if (target is None):
                final = self.adversarial_loss(perturbed_logits, self.y(shape_index))
            else:
                final = self.adversarial_loss_w_target(perturbed_logits, self.target(shape_index))
            
            summation += final
            sum_list.append(float(final))
        self._list = sum_list
        return summation


class UniversalL2Similarity(LossFunction):
    def __init__(self, data: BaseData, perturbation: UniversalPerturbation):
        super().__init__(data, perturbation)
    
    def __call__(self) -> torch.Tensor:
        sumL2 = 0
        for shape_index in range(self.shape_count):
            # Area element (i)
            #_, area_values = self.area(shape_index) # Using previous calculation see Eigen).
            area_values = self.area(shape_index)
            # Eigenvectos (i)
            eigvecs = self.eigenvectors(shape_index)
            # Norm
            product = eigvecs.matmul(self.alpha_i(shape_index)) 
            weighted = product * torch.sqrt(area_values.view(-1,1))
            sumL2 += weighted.norm(p="fro") # this reformulation uses the sub-gradient (hence ensuring a valid behaviour at zero)
        return sumL2

import utils.misc as misc

class UniversalLocalSimilarity(LossFunction):
    def __init__(self, data: BaseData, perturbation: UniversalPerturbation, k: int = 30):
        super().__init__(data, perturbation)
        self.neighborhood = k
        self.n = data.vertex_count

        nn_list = []
        for shape_index in range(self.shape_count):
            nn = misc.kNN(pos = data.pos(shape_index), 
                          edges = data.edges(shape_index), 
                          neighbors_num = self.neighborhood, 
                          cutoff = 5)
            nn_list.append(nn)
        self.nn = torch.stack(nn_list, dim = 2)
    
    def __call__(self) -> torch.Tensor:
        summation = 0
        for shape_index in range(self.shape_count):
            eigvecs_i = self.eigenvectors(shape_index)

            perturbation = eigvecs_i.matmul(self.alpha_i(shape_index))

            pos = self.pos(shape_index)
            perturbed_pos = pos + perturbation

            flat_kNN = self.nn[:,:,shape_index].view(-1)
            X = pos[flat_kNN].view(-1, self.neighborhood, 3) # shape [n*K*3]
            Xr = perturbed_pos[flat_kNN].view(-1, self.neighborhood, 3)
            dist = torch.norm(X - pos.view(self.n,1,3), p=2, dim=-1)
            dist_r = torch.norm(Xr - perturbed_pos.view(self.n,1,3), p=2, dim=-1)
            dist_loss = torch.nn.functional.mse_loss(dist, dist_r, reduction="sum")
            summation += dist_loss
        return summation

class UniversarlCrossL2Similarity(LossFunction):
    def __init__(self, data: BaseData, perturbation: UniversalPerturbation):
        super().__init__(data, perturbation)

    def __call__(self) -> torch.Tensor:
        summation = 0
        for shape_index in range(self.shape_count):
            eigvecs_i = self.eigenvectors(shape_index)
            perturbation = eigvecs_i.matmul(self.alpha_i(shape_index))
            pinv = torch.pinverse(self.template_eigvecs)
            projection = pinv.matmul(perturbation)
            norm = (projection - self.alpha).norm(p = "fro")
            summation += norm
        return summation


from utils.spectral import *
from utils.eigendecomposition import Eigsh_torch
import matplotlib.pyplot as plt 
import time

class UniversalIsospec(LossFunction):
    def __init__(self, data: BaseData, perturbation: UniversalPerturbation):
        super().__init__(data, perturbation)
        self.device = data.device
        

    def __call__(self) -> torch.Tensor:
        summation = 0
        
        self.mean_disp =  0
        for shape_index in range(self.shape_count):
            k = self.eigencount_isosp
            kup = self.eigencount_isosp

            eigvecs_i = self.eigenvectors(shape_index)
            eigvals_i = self.eigenvalues(shape_index) #[:kup]

            perturbation = eigvecs_i.matmul(self.alpha_i(shape_index))
            perturbed_pos = self.pos(shape_index) + perturbation
            
            perturbed_pos = perturbed_pos.to(torch.float64)
            
            W, _, A = calc_LB_FEM(perturbed_pos, self.faces(0), device=perturbed_pos.device)
            C = decomposition_torch(W, A)
            perturbed_evals = Eigsh_torch.apply(C.to_dense(), C.values(),
                                                C.indices(), kup).to(self.epsilon.device).float()
            
            target_evals = eigvals_i * (1 + self.epsilon)

            loss_i = torch.mean(((perturbed_evals[1:k] - target_evals[1:k])/eigvals_i[1:k]).pow(2))
            summation += loss_i
        return summation
