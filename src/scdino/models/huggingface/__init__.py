from transformers import AutoConfig, AutoModel

from scdino.models.huggingface.configuration_scdino import ScDINOConfig
from scdino.models.huggingface.modeling_scdino import ScDINOModel

AutoConfig.register("scdino", ScDINOConfig)
AutoModel.register(ScDINOConfig, ScDINOModel)

__all__ = ["ScDINOConfig", "ScDINOModel"]
