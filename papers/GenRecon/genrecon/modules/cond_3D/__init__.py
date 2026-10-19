"""3D conditioning modules: project multi-view image tokens onto a (dense or
sparse) voxel grid and aggregate them per voxel.

The dense and sparse variants share class names (``Projection``,
``MultiViewFeatAggregator``); import them from their submodules explicitly:

    from .cond_3D.projection import Projection                 # dense grid
    from .cond_3D.sparse_projection import Projection          # sparse tensor
    from .cond_3D.aggregation_net import MultiViewFeatAggregator        # dense
    from .cond_3D.sparse_aggregation_net import MultiViewFeatAggregator  # sparse
"""
