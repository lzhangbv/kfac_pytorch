from kfac.kfac_preconditioner_inv_dp import KFAC as KFAC_INV_DP
from kfac.kfac_preconditioner_eigen_dp import KFAC as KFAC_EIGEN_DP

def DP_KFAC(model, 
            inv_type='eigen', 
            lr=0.1,
            damping=0.001,
            fac_update_freq=1,
            kfac_update_freq=1,
            kl_clip=0.001,
            factor_decay=0.95,
            exclude_vocabulary_size=None,
            hook_enabled=True,
            exclude_parts=''):
    """
    DP_KFAC optimizer that wraps KFAC_INV_DP and KFAC_EIGEN_DP by setting inv_type. 
    """
    if inv_type == 'eigen':
        return KFAC_EIGEN_DP(model=model, 
                lr=lr, 
                damping=damping, 
                fac_update_freq=fac_update_freq,
                kfac_update_freq=kfac_update_freq,
                kl_clip=kl_clip,
                factor_decay=factor_decay,
                exclude_vocabulary_size=exclude_vocabulary_size,
                hook_enabled=hook_enabled, 
                exclude_parts=exclude_parts)
    else:
        return KFAC_INV_DP(model=model, 
                lr=lr, 
                damping=damping, 
                fac_update_freq=fac_update_freq,
                kfac_update_freq=kfac_update_freq,
                kl_clip=kl_clip,
                factor_decay=factor_decay,
                exclude_vocabulary_size=exclude_vocabulary_size,
                hook_enabled=hook_enabled, 
                exclude_parts=exclude_parts)

