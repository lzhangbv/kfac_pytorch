from kfac_refactor.kfac_preconditioner_base import KFACParamScheduler
from kfac_refactor.kfac_preconditioner_base import KFAC as KFAC_BASE
from kfac_refactor.kfac_preconditioner_eigen import KFAC as KFAC_EIGEN
from kfac_refactor.kfac_preconditioner_inv import KFAC as KFAC_INV

KFAC = KFAC_INV

kfac_mappers = {
    'base': KFAC_BASE,
    'eigen': KFAC_EIGEN,
    'inverse': KFAC_INV
    }

def get_kfac_module(kfac='inverse'):
    return kfac_mappers[kfac]
