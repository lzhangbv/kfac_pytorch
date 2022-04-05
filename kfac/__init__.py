from kfac.kfac_preconditioner_base import KFACParamScheduler
from kfac.kfac_preconditioner_inv import KFAC as KFAC_INV
from kfac.kfac_preconditioner_eigen import KFAC as KFAC_EIGEN
from kfac.kfac_preconditioner_inv_dp import KFAC as KFAC_INV_DP
from kfac.kfac_preconditioner_eigen_dp import KFAC as KFAC_EIGEN_DP
from kfac.dp_kfac import DP_KFAC 

kfac_mappers = {
    'inverse': KFAC_INV, 
    'eigen': KFAC_EIGEN,
    'inverse_dp': KFAC_INV_DP, 
    'eigen_dp': KFAC_EIGEN_DP,
    }

def get_kfac_module(kfac='eigen_dp'):
    return kfac_mappers[kfac]
