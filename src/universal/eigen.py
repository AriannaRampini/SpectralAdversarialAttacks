
import torch
import torch_geometric.data
import utils

class Eigen():
    def __init__(self, 
                 dataset: torch_geometric.data.InMemoryDataset, 
                 dtype: torch.dtype, 
                 isospectralization: int, 
                 lowband: int):
        self._isospectralization_num = isospectralization
        self._lowband_num = lowband 
        self._eigen_count = max(lowband, isospectralization)
        self._calculate(dataset, dtype)

    def _calculate(self, dataset: torch_geometric.data.InMemoryDataset, dtype: torch.dtype):
        r""" Calculating the count()+1 eigen decomposition, then discarding this first constant eigenvector."""
        # Save tensors on a list then stack them all up.
        eigen_values_list = []
        eigen_vectors_list = []

        self._stiff = []
        self._area = []

        for shape in dataset:
            W, _, A = utils.spectral.calc_LB_FEM(shape.pos, shape.face.t(), device = shape.pos.device)
            C = utils.spectral.decomposition_torch(W, A)

            numpy_eigvals, numpy_eigvectors = utils.spectral.eigsh(C.to_dense(), C.values(), C.indices(), self.count+1)

            A_inverse = A.rsqrt().detach().cpu().numpy()
            numpy_eigvectors_normalized = A_inverse[:,None] * numpy_eigvectors

            eigvals    = torch.tensor(numpy_eigvals, dtype=dtype, device=shape.pos.device, requires_grad=False)
            eigvectors = torch.tensor(numpy_eigvectors_normalized, dtype=dtype, device=shape.pos.device, requires_grad=False)

            #eigen_values_list.append(eigvals[1:])
            eigen_values_list.append(eigvals[:self._isospectralization_num])
            eigen_vectors_list.append(eigvectors[:,1:(self._lowband_num+1)])

            self._stiff.append(W)
            self._area.append(A)

        self._eigen_values = torch.stack(eigen_values_list, dim = 1)
        self._eigen_vectors = torch.stack(eigen_vectors_list, dim = 2)
        return

    @property
    def count(self):
        return self._eigen_count

    def eigenvalues(self, shape_index) -> torch.Tensor:
        return self._eigen_values[:, shape_index]

    def eigenvectors(self, shape_index) -> torch.Tensor:
        return self._eigen_vectors[:, :, shape_index]

    def stiff(self, shape_index) -> tuple: 
        return self._stiff[shape_index]

    def area(self, shape_index) -> tuple: 
        return self._area[shape_index]

    @property
    def isospectralization(self) -> int:
        return self._isospectralization
    
    @property
    def lowband(self) -> int:
        return self._lowband
    
