from genrecon import models

from ..modules.lora import apply_lora


def build_single_model(model_cfg):

    model = getattr(models, model_cfg.name)(**model_cfg.args)

    if hasattr(model_cfg, "lora") and model_cfg.lora is not None:
        model = apply_lora(model, model_cfg.lora)

    if hasattr(model_cfg, "projection") and model_cfg.projection is not None:
        use_camera = getattr(model_cfg.projection, "use_camera", False)
        refinement_factor = getattr(model_cfg.projection, "refinement_factor", 1)
        use_conv_net = getattr(model_cfg.projection, "conv_net", False)
        use_self_attention = getattr(model_cfg.projection, "self_attention", False)
        model.add_projections(
            model_cfg.projection.image_size,
            use_camera=use_camera,
            refinement_factor=refinement_factor,
            use_conv_net=use_conv_net,
            use_self_attention=use_self_attention,
        )

    return model
