
import torch
import torch_geometric.data
import utils
import numpy as np

from .base_data import BaseData
from .eigen import Eigen


class Data(BaseData):
    def __init__(self, dataset: torch_geometric.data.InMemoryDataset, isospectralization: int, lowband: int):
        super().__init__(dataset)

        self._dataset = dataset

        # Check for proper dimensions (right shape).
        utils.misc.check_data(self.pos(0), self.edges(0), self.faces(0), float_type=torch.float)

        self._vertex_count = dataset[0].pos.size()[0]
        self._shape_count = len(dataset)
        self._eigen = Eigen(dataset, self.dtype_float, isospectralization, lowband)

    @property
    def dtype_float(self):
        return torch.float64

    def pos(self, shape_index) -> torch.Tensor: 
        return self._dataset[shape_index].pos.to(self.dtype_float)
    
    def edges(self, shape_index) -> torch.LongTensor:
        return self._dataset[shape_index].edge_index.t()

    def faces(self, shape_index) -> torch.LongTensor:
        return self._dataset[shape_index].face.t()

    def y(self, shape_index) -> int:
        return self._dataset[shape_index].y

    @property
    def shape_count(self):
        return self._shape_count

    @property
    def vertex_count(self):
        return self._vertex_count
    
    @property
    def eigen_count(self):
        return self._eigen.count

    def eigenvalues(self, shape_index) -> torch.Tensor:
        return self._eigen.eigenvalues(shape_index)

    def eigenvectors(self, shape_index) -> torch.Tensor:
        return self._eigen.eigenvectors(shape_index)

    def stiff(self, shape_index) -> tuple: 
        return self._eigen.stiff(shape_index)

    def area(self, shape_index) -> tuple: 
        return self._eigen.area(shape_index)

    def file_name(self, shape_index) -> str:
        return self._dataset.file_name(shape_index)

    @property
    def isospectralization(self) -> int:
        return self._eigen._isospectralization_num
    
    @property
    def lowband(self) -> int:
        return self._eigen._lowband_num

    @property
    def eigen(self):
        r"""Return Eigen class object, useful to save all the eigenvalues and eigenvectors."""
        return self._eigen
