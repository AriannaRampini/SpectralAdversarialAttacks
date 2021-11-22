
import torch

from .misc import check_data

def write_off(pos: torch.Tensor,faces: torch.Tensor, file: str):
    check_data(pos = pos, faces = faces)
    n, m = pos.shape[0], faces.shape[0]
    pos = pos.detach().cpu().clone().numpy();
    faces = faces.detach().cpu().clone().numpy();

    file = file if file.split(".")[-1] == "off" else file + ".off" # add suffix if necessary
    with open(file, 'w') as f:
        f.write("OFF\n")
        f.write("{} {} 0\n".format(n, m))
        for v in pos:
            f.write("{} {} {}\n".format(v[0], v[1], v[2]))
            
        for face in faces:
            f.write("3 {} {} {}\n".format(face[0],face[1],face[2]))

def read_off(str_file: str):
    file = open(str_file, 'r')
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    pos   = torch.Tensor([[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)])
    faces = torch.Tensor([[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)])
    return pos, faces
