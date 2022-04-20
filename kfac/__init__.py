from kfac.fast_kfac import KFAC as FAST_KFAC 
from kfac.fast_kfac import KFACParamScheduler

kfac_mappers = {
    'fast': FAST_KFAC,
    }

def get_kfac_module(kfac='fast'):
    return kfac_mappers[kfac]
