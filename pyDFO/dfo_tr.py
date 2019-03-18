"""This is a derivative free trust region optimization algorithm. 
This algorithm is designed to minimize blackbox functions.

Written by Liyuan Cao @Lehigh University in 2018, under the supervision 
of Dr. Katya Scheinberg. 
"""

import numpy as np
import time
from param import param
from sample import sample
from build_model import build_model
from trust_sub import trust_sub
from trust_region_update import trust_region_update


def dfo_tr(func, x, customizedPar):
    # start timing 
    start_time = time.time()

    # initializing parameters
    n = x.shape[0]
    par = param(n)
    par.overwrite(customizedPar)

    # counters: iteration, successful iteration and function evaluation
    iteration = iter_suc = func_eval = 0

    # build initial sample
    samp = sample(x, func, par)
    f = samp.f[-1]
    func_eval += samp.m
    
    # print out the initial evaluation results
    if par.verbosity >= 2:
        print ("\n Iteration Report ")
        print ('|  iter |suc|  objective  | TR_radius |    rho    | m  | mgood')
        print ("| {:5d} |---| {:11.5e} | {:9.6f} | --------- | {} ".format(iteration, f, par.tr_delta, samp.m))

    # main iterative algorithm
    while iteration <= par.stop_iter:

        # build an approximation model
        model = build_model(samp, par)

        # trust region subproblem
        s,val = trust_sub(x - samp.center, model, par)

        # check out the new point
        x1 = x + s      # the new point
        m1 = f + val    # model evaluated at the new point
        f1 = func(x1)   # function evaluated at the new point
        func_eval += 1
        
        # calculate the ratio between real reduction in function value and 
        # the reduction in the value of the approximating model
        rho  = (f - f1) / (f - m1)

        # evaluate the quality of the new point and update the trust region 
        success, par = trust_region_update(rho, samp.mgood, par)

        # go to the new point if succeed
        iteration += 1
        if success:
            iter_suc += 1
            x = x1
            f = f1


        # print iteration report
        if par.verbosity >= 2:
            print ("| {:5d} | {} | {:11.5e} | {:9.6f} | {:9.6f} | {} | {} |"
            .format(iteration, success, float(f), float(par.tr_delta), float(rho), samp.m, samp.mgood))

        # check stopping criteria
        # Stopping criterion -- stop if norm of g is too small
        normg = np.linalg.norm(model.g)
        if normg <= par.stop_g and not success:
            break
        # Stopping criterion -- stop if the TR radius is too small
        if par.tr_delta < par.stop_delta:
            break

        # update sample
        samp.update(x, f, x1, f1, par, iteration)

        if par.alg_poised:
            # poisedness improvement sample 
            points = samp.poisedness_improvement_points(1, par)
            func_eval += 1

            # evaluate poisedness improvement sample points
            fvalues = np.array( map(func, points) )

            # add points to the sample
            samp.addpoints(points, fvalues, par)

    # end timing
    end_time = time.time()

    # final report
    if par.verbosity >= 1:
        print ('***************** FINAL REPORT ************************\n')
        print ("Total time is {} seconds.\n".format(end_time - start_time))
        print ("Norm of the gradient of the model is {:.20f}.\n".format(normg))
        print ("|iter | #success| #fevals| final fvalue | final tr_radius|\n")
        print ("| {} |    {}   |   {}   |   {}   |  {}  \n"
                .format(iteration, iter_suc, func_eval, f, par.tr_delta))

    
    return x, f

