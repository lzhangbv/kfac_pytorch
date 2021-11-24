from kfac.kfac_preconditioner_eigen import KFACParamScheduler
from kfac.kfac_preconditioner_eigen import KFAC as KFAC_EIGEN
from kfac.kfac_preconditioner_inv import KFAC as KFAC_INV
from kfac.kfac_preconditioner_inv_nopar import KFAC as KFAC_INV_NOPAR
from kfac.kfac_preconditioner_inv_opt import KFAC as KFAC_INV_OPT
from kfac.kfac_preconditioner_inv_nordc import KFAC as KFAC_INV_NORDC
from kfac.kfac_preconditioner_inv_rdc import KFAC as KFAC_INV_RDC
from kfac.kfac_preconditioner_inv_rdc_ppl import KFAC as KFAC_INV_RDC_PPL

KFAC = KFAC_INV

kfac_mappers = {
    'eigen': KFAC_EIGEN,
    'inverse': KFAC_INV,
    'inverse_nopar': KFAC_INV_NOPAR,
    'inverse_opt': KFAC_INV_OPT,
    'inverse_nordc': KFAC_INV_NORDC,
    'inverse_rdc': KFAC_INV_RDC,
    'inverse_rdc_ppl': KFAC_INV_RDC_PPL
    }

def get_kfac_module(kfac='eigen'):
    return kfac_mappers[kfac]
