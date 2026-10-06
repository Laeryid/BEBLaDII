import torch

class Mesh:
    def __init__(self, devices, mesh_shape, axis_names=None):
        self.devices = devices
        self.mesh_shape = mesh_shape
        self.axis_names = axis_names

def mark_sharding(tensor, mesh, partition_spec):
    return tensor

def set_global_mesh(mesh):
    pass
