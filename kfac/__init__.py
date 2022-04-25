from kfac.fast_kfac import KFAC as FAST_KFAC 
from kfac.fast_shampoo import KFAC as FAST_SHAMPOO
from kfac.fast_kfac import KFACParamScheduler

kfac_mappers = {
    'fast': FAST_KFAC,
    'shampoo': FAST_SHAMPOO
    }

def get_kfac_module(kfac='fast'):
    return kfac_mappers[kfac]
