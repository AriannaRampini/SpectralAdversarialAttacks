
import torch

from .base_data import BaseData

class Perturbation(object):
  def __init__(self, data: BaseData):
    super().__init__()
    self._data = data
    self._alpha = None
    self._perturbed_pos_cache = None
  
  @property
  def data(self): 
    return self._data

  def reset(self):
    self._reset()
    self._perturbed_pos_cache = None
    def hook(grad): self._perturbed_pos_cache = None
    self.alpha.register_hook(hook)

  def perturb_positions(self):
    if self._perturbed_pos_cache is None:
      self._perturbed_pos_cache = self._perturb_positions()
    return self._perturbed_pos_cache

  def variables_to_optimize(self):
    raise NotImplementedError

  def save_last(self):
    return NotImplementedError

  def restore_last(self):
    return NotImplementedError

class UniversalPerturbation(Perturbation):
  def __init__(self, data: BaseData, targets: [int], template_index = 0):
    super().__init__(data)
    # Extra deformation parameter to optimize.
    self._alpha_i = None
    self._epsilon = None

    self._template_index = template_index
    self._template_eigvals = self.data.eigenvalues(template_index)
    self._template_eigvecs = self.data.eigenvectors(template_index)

    # Sanity check.
    assert(targets is None or len(targets) == data.shape_count)

    self._targets = targets

    self.reset()

    self._last_alpha   = torch.Tensor()
    self._last_alpha_i = torch.Tensor()
    self._last_epsilon = torch.Tensor()

  def variables_to_optimize(self):
    return [self._alpha, self._alpha_i, self._epsilon]

  def is_alpha_i(self):
    return True

  def save_last(self):
    self._last_alpha.data = self._alpha.data.clone()
    self._last_alpha_i.data = self._alpha_i.data.clone()
    self._last_epsilon.data = self._epsilon.data.clone()
    return

  def restore_last(self):
    self.reset()
    self._alpha.data = self._last_alpha
    self._alpha_i.data = self._last_alpha_i
    self._epsilon.data = self._last_epsilon
    return

  def _reset(self):
    self._alpha: torch.Tensor = torch.zeros([self.data.lowband, 3], 
                                             device = self.data.device, 
                                             dtype = self.data.dtype_float, 
                                             requires_grad = True)
    self._alpha_i: torch.Tensor = torch.zeros([self.data.lowband, 3, self.data.shape_count], 
                                               device = self.data.device, 
                                               dtype = self.data.dtype_float, 
                                               requires_grad = True)
    self._epsilon: torch.Tensor = torch.zeros([self.data.isospectralization], 
                                               device = self.data.device, 
                                               dtype = self.data.dtype_float, 
                                               requires_grad = True)

  def _perturb_positions(self):
    return self.data.pos(self.template_index) + self.template_eigvecs.matmul(self.alpha)

  def perturbation_alpha(self):
    r"""Returns perturbation through alpha."""
    return self.template_eigvecs.matmul(self.alpha)

  def perturbation_alpha_i(self, shape_index):
    r"""Returns perturbation through alpha_i with corresponding base."""
    return self.data.eigenvectors(shape_index).matmul(self.alpha_i[:,:,shape_index])

  def perturb_positions_i(self, shape_index):
    r"""Return perturbed positions for a given shape via alpha_i and correspoding base."""
    return self.data.pos(shape_index) + self.perturbation_alpha_i(shape_index)

  @property
  def alpha(self): return self._alpha

  @property
  def alpha_i(self):
      return self._alpha_i
  
  @property
  def epsilon(self):
    return self._epsilon

  @property
  def targets(self):
      return self._targets

  @property
  def template_eigvecs(self): 
    return self._template_eigvecs
  
  @property
  def template_eigvals(self): 
    return self._template_eigvals

  @property
  def template_index(self):
      return self._template_index
