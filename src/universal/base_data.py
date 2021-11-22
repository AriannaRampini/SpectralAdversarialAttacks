
import torch
import torch_geometric.data

class BaseData():
    """ Encapsulate interface to interact with the dataset. This way we have the flexibility of accesing one shape or the whole dataset in our case of universal attacks.
    """
    def __init__(self, dataset: torch_geometric.data.InMemoryDataset):
        self._device: torch.device = dataset[0].pos.device
        self._dtype_int: torch.dtype = dataset[0].edge_index.dtype
        self._dtype_float: torch.dtype = dataset[0].pos.dtype

        self._vertex_count = dataset[0].pos.shape[0]
        self._edge_count = dataset[0].edge_index.t().shape[0]
        self._face_count = dataset[0].face.t().shape[0]

    @property
    def device(self):
        return self._device
    
    @property
    def dtype_int(self):
        return self._dtype_int
    
    @property
    def dtype_float(self):
        return self._dtype_float 
    
    @property
    def vertex_count(self) -> int: 
        return self._vertex_count
    
    @property
    def edge_count(self) -> int: 
        return self._edge_count
    
    @property
    def face_count(self) -> int: 
        return self._face_count

    def pos(self) -> torch.Tensor: 
        raise NotImplementedError()
    
    def edges(self) -> torch.LongTensor:
        raise NotImplementedError()

    def faces(self) -> torch.LongTensor:
        raise NotImplementedError()
