from ..experimental.xla_sharding import mark_sharding, Mesh
from ..experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel

class xla_sharding:
    mark_sharding = staticmethod(mark_sharding)
    Mesh = Mesh
