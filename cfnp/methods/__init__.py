from cfnp.methods.base import BaseModel, ClassSeparationBaseModel, AddInstanceBaseModel

# from cfnp.methods.gp import CompressionNetForGP
# from cfnp.methods.klr import CompressionNetForKLR
# from cfnp.methods.knn import CompressionNetForKNN
# from cfnp.methods.knr import CompressionNetForKNR
# from cfnp.methods.krr import CompressionNetForKRR
from cfnp.methods.svc import create_svc_class
# from cfnp.methods.svr import CompressionNetForSVR

BASES = {
    'base': BaseModel,
    'sep': ClassSeparationBaseModel,
    'add': AddInstanceBaseModel
}

METHODS = {
    # 'gp': CompressionNetForGP,
    # 'klr': CompressionNetForKLR,
    # 'knn': CompressionNetForKNN,
    # 'knr': CompressionNetForKNR,
    # 'krr': CompressionNetForKRR,
    'svc': create_svc_class,
    # 'svr': CompressionNetForSVR
}

__all__ = [
    # 'CompressionNetForGP',
    # 'CompressionNetFroKLR',
    # 'CompressionNetForKNN',
    # 'CompressionNetForKNR',
    # 'CompressionNetForKRR',
    'create_svc_class',
    # 'CompressionNetForSVR'
]