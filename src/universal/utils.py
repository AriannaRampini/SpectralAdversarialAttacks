
import torch

from .adversarial import UniversalAdversarialExample
from .eigen       import Eigen
from .parameters  import Parameters, ParametersEncoder

def save(adversarial: UniversalAdversarialExample, str_path: str):
    r"""Saving some results to be able to load them later and reproduce the perturbations.
        Extremely simple for now. Save alpha, alpha_i and all eigenvalues/vectors.
     """
    tosave = {
        'epsilon'     : adversarial.perturbation.epsilon,
        'alpha_i'     : adversarial.perturbation.alpha_i,
        'eigen'       : adversarial.data.eigen,
        'eigenvectors': adversarial.data.eigen._eigen_vectors,
        'n_shapes'    : adversarial.shape_count(),
        'targets'     : adversarial.target
    }

    torch.save(tosave, str_path)
    return

def load(str_path):
    r"""Load saved results."""
    return torch.load(str_path)


import json

def save_params(parameters: Parameters, str_path: str):
    r"""Save parameters in binary form to then load them. Also save them in 
        human readable form just to easily look at them. 
     """
    filehandler = open(str_path, 'w') 
    # pickle.dump(parameters, filehandler)
    filehandler.write(json.dumps(parameters, indent=4, cls=ParametersEncoder))
    filehandler.close()
    return

def load_params(str_path: str): 
    filehandler = open(str_path, 'r') 
    params = Parameters()
    params.__dict__ = json.load(filehandler)
    return params

def save_dic(dic_to_save: dict(), str_path: str):
    r"""Save losses in binary form to then load them. Also save them in 
        human readable form just to easily look at them. 
     """
    filehandler = open(str_path, 'w') 
    filehandler.write(json.dumps(dic_to_save, indent=4))
    filehandler.close()
    return

def save_list(list_to_save: list, str_path: str):
    r"""Save list of indices of shapes used in binary form to then load it. 
        Also save it in human readable form just to easily look at it. 
     """
    filehandler = open(str_path, 'w') 
    filehandler.write(json.dumps(list_to_save))
    filehandler.close()
    return


def load_json(str_path: str): 
    filehandler = open(str_path, 'r') 
    out_data = json.load(filehandler)
    return out_data