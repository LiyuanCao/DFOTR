
def trust_region_update(rho, mgood, par):

    # Updating iterate and trust-region radius.
    if rho >= par.tr_toaccept:
        # When the approximation is sufficiently good, 
        # accept the new point.
        success = 1

        # When the approximation is very good, increase TR radius
        if rho >= par.tr_toexpand:
            par.tr_delta = min(par.tr_expand * par.tr_delta, 1e3)

    else:
        # When the approximation is bad, function value is not 
        # significantly reduced, reject the new point.
        success = 0
    
        # reduce TR radius
        if mgood >= par.tr_toshrink:
            par.tr_delta *= par.tr_shrink 

    return success, par