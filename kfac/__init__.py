from kfac.kfac_preconditioner_eigen import KFACParamScheduler
from kfac.kfac_preconditioner_eigen import KFAC as KFAC_EIGEN
from kfac.kfac_preconditioner_inv import KFAC as KFAC_INV
from kfac.kfac_preconditioner_inv_opt import KFAC as KFAC_INV_OPT

KFAC = KFAC_INV

kfac_mappers = {
    'eigen': KFAC_EIGEN,
    'inverse': KFAC_INV,
    'inverse_opt': KFAC_INV_OPT
    }

def get_kfac_module(kfac='eigen'):
    return kfac_mappers[kfac]
