"""The Solv3R network. The image encoder, the output heads and the camera conversions are adapted from DVLT
(https://github.com/nv-tlabs/dvlt); each file names its sources."""
from .network import Solv3RNetwork

__all__ = ["Solv3RNetwork"]
