from os import listdir, mkdir
from os.path import isfile, join, exists, split
import tarfile

import tqdm
import torch
import torch_geometric.data
import torch_geometric.transforms as transforms
import openmesh
import torch_geometric.io as gio

import dataset.downscale as dscale
from utils.transforms import Move, Rotate, ToDevice


class SmalDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, 
        root:str, 
        device:torch.device=torch.device("cpu"),
        train:bool=True, 
        test:bool=True,
        custom:bool=False,
        custom_list:[int] = [],
        transform_data:bool=True):

        self.categories = ["big_cats","cows","dogs","hippos","horses"]

        # center each mesh into its centroid
        pre_transform = transforms.Center()

        # transform
        if transform_data:
            # rotate and move
            transform = transforms.Compose([
                Move(mean=[0,0,0], std=[0.05,0.05,0.05]), 
                Rotate(dims=[0,1,2]), 
                ToDevice(device)])
        else:
            transform=ToDevice(device)

        super().__init__(root=root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.downscaler = dscale.Downscaler(
            filename=join(self.processed_dir,"ds"), mesh=self.get(0), factor=2)

        if train and not test and not custom:
            self.mapping = [i for i in range(len(self)) if self.get(i).pose < 16]
            self.subset  = list(range(len(self.mapping))) 
            self.data, self.slices = self.collate([self.get(i) for i in range(len(self)) if self.get(i).pose < 16])
        elif not train and test and not custom:
            self.mapping = [i for i in range(len(self)) if self.get(i).pose >= 16]
            self.subset  = list(range(len(self.mapping))) 
            self.data, self.slices = self.collate([self.get(i) for i in range(len(self)) if self.get(i).pose >= 16])
        elif not train and not test and custom:
            possible_indices = [i for i in range(len(self)) if self.get(i).pose >= 16] # Whole test set.
            self.mapping = [possible_indices[i] for i in custom_list]
            self.subset  = custom_list
            self.data, self.slices = self.collate([self.get(possible_indices[i]) for i in custom_list])

    @property
    def raw_file_names(self):
        files = sorted(listdir(self.raw_dir))
        categ_files = [f for f in files if isfile(join(self.raw_dir, f)) and f.split(".")[-1]=="ply"]
        return categ_files

    def file_name(self, index):
        r"""Returns file name by mapping its index within the dataset to its original name."""
        assert(index in range(len(self.mapping)))
        map = lambda x: self.raw_file_names[self.mapping[x]] 
        return map(index).split(".")[0]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found.')
    
    def process(self):
        data_list = []
        f2e = transforms.FaceToEdge(remove_faces=False)
        for pindex, path in enumerate(tqdm.tqdm(self.raw_paths)):
            mesh = gio.read_ply(path)
            f2e(mesh)
            tmp = split(path)[1].split(".")[0].split("_")
            model_str, pose_str = tmp[-2], tmp[-1]
            category = "_".join(tmp[:-2])
            mesh.model = int(model_str[5:])
            mesh.pose = int(pose_str[4:])
            mesh.y = self.categories.index(category)
            if self.pre_filter is not None and not self.pre_filter(mesh) : continue
            if self.pre_transform is not None: mesh = self.pre_transform(mesh)
            data_list.append(mesh)
        data, slices = self.collate(data_list)
        torch.save( (data, slices), self.processed_paths[0])

    @property
    def downscale_matrices(self): return self.downscaler.downscale_matrices

    @property
    def downscaled_edges(self): return self.downscaler.downscaled_edges

    @property
    def downscaled_faces(self): return self.downscaler.downscaled_faces
