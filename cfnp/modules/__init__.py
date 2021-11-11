from cfnp.modules.conv import ConvCompressionModule
from cfnp.modules.gumble import GumbleCompressionModule


MODULES= {
    'conv': ConvCompressionModule,
    'gumble': GumbleCompressionModule,
}

__all__ = [
    'ConvCOmpressionModule',
    'GumbleCOmpressionMOdule',
]