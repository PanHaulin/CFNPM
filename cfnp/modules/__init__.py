from cfnp.modules.conv import ResCompressionNet
from cfnp.modules.gumble import GumbleCompressionModule


MODULES= {
    'resnet18': ResCompressionNet,
    'resnet34': ResCompressionNet,
    'resnet50': ResCompressionNet,
    'gumble': GumbleCompressionModule,
}

__all__ = [
    'ResCompressionNet',
    'GumbleCompressionModule',
]