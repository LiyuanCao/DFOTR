import numpy as np
import sys
import time
from .Sample import Sample
from .ApproximationModel import ApproximationModel

class dfo_tr:
    """ This is a derivative free trust region optimization algorithm. 
        This algorithm is designed to minimize blackbox functions.

        Written by Liyuan Cao @Lehigh University in 2020, under the supervision 
        of Dr. Katya Scheinberg. 
    """

    def __init__(self, x0, options=None):
        # input dimension
        self.n = len(x0)

        # default options
        self.options = {
            # the algorithm
            'alg_model': 'quadratic',   # the model type
            'alg_TR': 'ball', # trust region type
            'alg_TRsub': 'exact',         # trust region subproblem
            # sample management parameters
            'sample_initial': self.n + 1,   # initial sample size
            'sample_min': int((self.n+1) * 1.1), # minimum sample size
            'sample_max': min(3000, (self.n+1)*(self.n+2)//2), # maximum sample size
            'sample_toremove': 100, # a point is to be removed from the sample set, if it is sample_toremove * tr_delta away from the current point
            # trust region parameters
            'tr_delta': 1.0,    # initial delta (i.e. trust region radius)
            'tr_toaccept': 0.0, # rho level to accept new point
            'tr_toexpand': 0.5, # rho level to expand radius
            'tr_expand': 1.3,   # radius expansion factor
            'tr_toshrink': -5e-3, 
            'tr_shrink': 0.95,  # radius shrink factor
            # stopping crieria parameters
            'stop_iter': 2000,  # maximum number of iterations
            'stop_nfeval': 2000, # maximum number of function evaluations
            'stop_delta': 1e-6, # throshold for trust region radius
            'stop_predict': 1e-8,    # threshold for predicted decrease
            # verbosity
            'verbosity': 2
        }

        # Override default options with custom ones.
        if options:
            for key in options:
                if key not in self.options.keys():
                    raise ValueError("{!r} is not a valid option name. \n".format(key))
                self.options[key] = options[key]

        # information/logger
        self.info = {}
        self.info['start_time'] = time.time()
        self.info['iteration'] = 0
        self.info['success'] = 0
        self.info['nfeval'] = 0

        # Create initial sample. 
        self.samp = Sample(x0, self.options)

        # Initialize model. 
        self.model = ApproximationModel(self.n, self.options)

    def ask(self, nAsk=1):
        assert nAsk == 1, 'The input nAsk must be 1.'\
            ' It can be larger than 1 in future versions of this software. '

        # Find the first not evaluated point in the sample set.
        idx = np.isnan(self.samp.fY).argmax()

        return [self.samp.Y[idx]]

    def tell(self, X, fX):
        assert np.all(X == X[0]), \
            'All the evaluated points in one batch must be the same point. '
        
        # Store the newly acquired function value. 
        idx = np.all(self.samp.Y==X[0], axis=1).argmax()
        self.samp.fY[idx] = np.mean(fX)
        self.info['nfeval'] += 1

        # If there is still a point that is not evaluated, we go to evaluate it. 
        if np.any(np.isnan(self.samp.fY)):
            return

        # If every point is evaluated, we begin the optimization process.
        self.__suggest()

    def __suggest(self):
        'Run one DFO-TR iteration.'

        if self.info['iteration'] == 0:
            # Put center of the initial trust region at the point in the
            #  initial sample with the lowest function value. 
            self.model.center = self.samp.Y[self.samp.fY.argmin()]
            self.model.delta = self.options['tr_delta']

            if self.options['verbosity'] >= 2:
                print("\n Iteration Report ")
                print('|  iter |suc|  objective  | TR_radius |    rho    | m  |')
                print("| {:5d} |---| {:11.5e} | {:9.6f} | --------- | {} "\
                    .format(self.info['iteration'],
                            min(self.samp.fY),
                            self.model.delta,
                            self.samp.m
                            ))

        else: 
            # Calculate the ratio between the actual reduction in function 
            # value and the reduction predicted the approximation model. 
            rho = (self.model.c - self.samp.fY[-1]) / self.info['predicted_decrease']

            # Calculate the ratio between the step size and the trust region radius.
            stepSize = np.linalg.norm(self.samp.Y[-1] - self.model.center)
            stepSize2delta = stepSize / self.model.delta

            # Decide whether to move the iterate. 
            if rho >= self.options['tr_toaccept']:
                self._success = 1
                self.info['success'] += 1
                self.model.center = self.samp.Y[-1]
            else:
                self._success = 0

            # Update the trust region radius. 
            self.model.update_delta(rho, stepSize2delta, self.options)

            # Remove points that are too far away from the current TR. 
            self.samp.auto_delete(self.model, self.options)

            # print iteration report
            if self.options['verbosity'] >= 2:
                print("| {:5d} | {} | {:11.5e} | {:9.6f} | {:9.6f} | {} |" \
                    .format(self.info['iteration'],
                            self._success,
                            self.samp.fY[-1],
                            self.model.delta, 
                            rho, 
                            self.samp.m
                            ))

        # build an approximation model
        self.model.fit(self.samp)

        # Solve the trust region subproblem. 
        x1, self.info['predicted_decrease'] = self.model.minimize()

        # Add the new point to the sample set. 
        self.samp.addpoint(x1)      # the new point
        self.info['iteration'] += 1

    def _stop(self):
        STOP = False
        if self.info['iteration'] == 0:
            return STOP

        if self.info['iteration'] >= self.options['stop_iter']:
            STOP = True
            print('Exiting because the maximum number of iterations is reached.')
        elif self.info['nfeval'] >= self.options['stop_nfeval']:
            STOP = True
            print('Exiting because the maximum number of function evaluations is reached.')
        elif self.model.delta <= self.options['stop_delta']:
            STOP = True
            print('Exiting because the minimum trust region radius is reached.')
        elif self.info['predicted_decrease'] <= self.options['stop_predict']:
            STOP = True
            print('Exiting because the minimum predicted decrease is reached.')
        
        if STOP and self.options['verbosity'] >= 1:
            print('***************** FINAL REPORT ************************')
            self.info['end_time'] = time.time()
            print('total elapsed time: {} seconds\n'.format(self.info['end_time'] - self.info['start_time']))
            print("|#iter|#success|#fevals| best fvalue |final tr_radius|final predicted decrease|")
            print("|{:5d}| {:5d}  | {:5d} | {:11.5e} |   {:9.6f}   |       {:11.5e}      |\n"
                .format(self.info['iteration'],
                        self.info['success'],
                        self.info['nfeval'], 
                        min(self.samp.fY), 
                        self.model.delta,
                        self.info['predicted_decrease']
                        ))

        return STOP



    @classmethod
    def optimize(cls, obj, x0, options=None):
        optimizer = dfo_tr(x0, options)

        while True:
            x = optimizer.ask()
            fx = [obj(x[0])]
            optimizer.tell(x,fx)
            if optimizer._stop():
                break
        idx = np.nanargmin(optimizer.samp.fY)
        return optimizer.samp.Y[idx], optimizer.samp.fY[idx], optimizer.info