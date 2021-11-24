from cfnp.baselines.prototype_generation import PrototypeGeneration
from cfnp.baselines.prototype_selection import PrototypeSelection
from cfnp.baselines.stratified_sampling import StratifiedSampling

DATASET_BASED_BASELINES_FOR_CLASSIFICATION = {
    'prototype_generation': PrototypeGeneration,
    'prototype_selection': PrototypeSelection,
}

DATASET_BASED_BASELINES_FOR_REGRESSION = {
    # TODO: 分层采样
    'stratified_sampling': StratifiedSampling
}

SVC_BASED_BASELINES = {
    
}