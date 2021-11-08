from kfac_refactor.kfac_preconditioner_base import KFACParamScheduler
from kfac_refactor.kfac_preconditioner_base import KFAC as KFAC_BASE
from kfac_refactor.kfac_preconditioner_inv import KFAC as KFAC_INV
from kfac_refactor.kfac_preconditioner_eigen import KFAC as KFAC_EIGEN
from kfac_refactor.kfac_preconditioner_inv_dp import KFAC as KFAC_INV_DP
from kfac_refactor.kfac_preconditioner_inv_dp_block import KFAC as KFAC_INV_DP_BLOCK
from kfac_refactor.kfac_preconditioner_eigen_dp import KFAC as KFAC_EIGEN_DP

KFAC = KFAC_INV

kfac_mappers = {
    'base': KFAC_BASE,
    'inverse': KFAC_INV, 
    'eigen': KFAC_EIGEN,
    'inverse_dp': KFAC_INV_DP, 
    'eigen_dp': KFAC_EIGEN_DP,
    'inverse_dp_block': KFAC_INV_DP_BLOCK
    }

def get_kfac_module(kfac='inverse'):
    return kfac_mappers[kfac]
