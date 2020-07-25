"""This file executes the derivative free optimization (DFO) solver to 
minimize a blackbox function.
 
User is required to import a blackbox function, and provide a 
starting point. User can overwrite the default algorithm and/or the 
default parameters used in the solver. 
"""

import numpy as np
# import solver
from Python2_obsolete.dfo_tr import dfo_tr
# import blackbox functions
from funcs_def import arwhead, rosen


# choose function
# func = arwhead
func = rosen

# starting point
# x0 = np.ones(200) * 1
x0 = np.repeat(np.array([[-1.2, 1]]), 10, axis=0).flatten()

# overwrite default settings
customizedPar = {'alg_model': 'linear', 'alg_TRS': 'exact', "stop_iter": 10000, "tr_delta": 1.0, 
                "verbosity": 2, "sample_toremove": 30, "sample_initial": 2}
# customizedPar = {'alg_model': 'quadratic', 'alg_TRS': 'exact', "stop_iter": 10000, "tr_delta": 1.0, 
#                 "verbosity": 2, "sample_toremove": 30}

# solve
x,f = dfo_tr(func, x0, customizedPar)

# print result
print("Printing result for function " + func.__name__ + ":")
print("best point: {}, with obj: {:.6f}".format(
    np.around(x, decimals=5), float(f)))
