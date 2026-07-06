from .pipeline_wan_fun_control import WanFunControlPipeline
from .pipeline_geoworld_stage1 import WanFunControlPipelineGeoWorldstage1
from .pipeline_geoworld_stage2 import WanFunControlPipelineGeoWorldstage2

import importlib.util

if importlib.util.find_spec("pai_fuser") is not None:
    from pai_fuser.core import sparse_reset

    WanFunControlPipelineGeoWorldstage1.__call__ = sparse_reset(WanFunControlPipelineGeoWorldstage1.__call__)
    WanFunControlPipelineGeoWorldstage2.__call__ = sparse_reset(WanFunControlPipelineGeoWorldstage2.__call__)
