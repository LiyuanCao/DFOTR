import numpy as np
import copy
import sys
import time
from .Sample import Sample
from .ApproximationModel import ApproximationModel
from .util import phi, qr, alpha2cgH

class dfo_tr:
    """ This is a derivative free trust region optimization algorithm. 
        This algorithm is designed to minimize blackbox functions.

        Written by Liyuan Cao @Lehigh University in 2020, under the supervision 
        of Dr. Katya Scheinberg. 
    """

    def __init__(self, x0, bounds=None, options=None):
        # input dimension
        self.n = len(x0)

        # bounds
        if bounds:
            assert len(bounds) == self.n, \
                'Dimension of the bounds does not match dimension of the input. '
        self.bounds = bounds

        # default options
        self.options = {
            # the algorithm
            'alg_model': 'quadratic',   # the model type
            'alg_TR': 'ball', # trust region type
            'alg_TRsub': 'exact',         # trust region subproblem
            'alg_feval': 1, # number of evaluation for each point
            # sample management parameters
            'sample_initial': self.n + 1,   # initial sample size
            'sample_min': int((self.n+1) * 1.1), # minimum sample size
            'sample_max': min(300, (self.n+1)*(self.n+2)//2), # maximum sample size
            'sample_toremove': 100, # a point is to be removed from the sample set, if it is sample_toremove * tr_delta away from the current point
            # trust region parameters
            'tr_delta': 1.0,    # initial delta (i.e. trust region radius)
            'tr_toaccept': 0.0, 
            'tr_toexpand': 0.5, # rho level to expand radius
            'tr_expand': 1.3,   # radius expansion factor
            'tr_toshrink': -5e-3, 
            'tr_shrink': 0.7**(1.0/np.sqrt(self.n)),  # radius shrink factor
            # stopping crieria parameters
            'stop_iter': 2000,  # maximum number of iterations
            'stop_nfeval': 2000, # maximum number of function evaluations
            'stop_delta': 1e-6, # throshold for trust region radius
            'stop_predict': 1e-8,    # threshold for predicted decrease
            # rng seed
            'rng_seed': None,
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
        self.info['x1success'] = 0
        self.info['x2success'] = 0
        self.info['nfeval'] = 0

        # rng
        if self.options['rng_seed']:
            np.random.seed(self.options['rng_seed'])

        # Create initial sample. 
        self.samp = Sample(x0, self.bounds, self.options)

        # Initialize model. 
        self.model = ApproximationModel(self.n, self.bounds, self.options)

    def ask(self, nAsk=1):
        # Find the first nAsk not evaluated points in the sample set.
        idx = np.isnan(self.samp.fYeval).any(axis=1).nonzero()[0][:nAsk]
        # assert len(idx) == nAsk, 'Insufficient points returned from ask. '
        if len(idx) != nAsk:
            print(self.samp.Y)
            print(self.samp.fYeval)
            print(idx)
            raise('Insufficient points returned from ask. ')
        idx = np.repeat(idx, self.options['alg_feval'])
        return self.samp.Y[idx]

    def tell(self, X, fX):
        # Store the newly acquired function value. 
        for i in range(len(X)):
            idx = np.all(self.samp.Y==X[i], axis=1).argmax()
            firstnan = np.isnan(self.samp.fYeval[idx]).argmax()
            self.samp.fYeval[idx, firstnan] = fX[i]
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
                print('|  iter |suc|    f(x1)    |    f(x2)    | TR_radius |    rho    | m  |')
                print("| {:5d} |---| {:11.5e} | {:11.5e} | {:9.6f} | --------- | {} "\
                    .format(self.info['iteration'],
                            self.samp.fY[0],
                            self.samp.fY[1],
                            self.model.delta,
                            self.samp.m
                            ))

        else: 
            x1idx = np.all(self.samp.Y == self.x1, axis=1).argmax()
            x2idx = np.all(self.samp.Y == self.x2, axis=1).argmax()

            if self.info['predicted_decrease'] > 0:
                # If the last iteration has an optimization step, 
                # calculate the ratio between the actual reduction in function 
                # value and the reduction predicted by the approximation model. 
                rho = (self.model.c - self.samp.fY[x1idx]) / self.info['predicted_decrease']

                # Calculate the ratio between the step size and the trust region radius.
                if self.options['alg_TR'] is 'ball':
                    stepSize = np.linalg.norm(self.samp.Y[x1idx] - self.model.center)
                elif self.options['alg_TR'] is 'box':
                    stepSize = np.linalg.norm(self.samp.Y[x1idx] - self.model.center, np.inf)
                stepSize2delta = stepSize / self.model.delta

                if self.samp.fY[x2idx] == np.min(self.samp.fY):
                    # Decide whether to move the iterate to x2. 
                    self._success = 2
                    self.info['x2success'] += 1
                    self.model.center = self.samp.Y[x2idx]
                elif rho >= self.options['tr_toaccept']:
                    # Decide whether to move the iterate to x1. 
                    self._success = 1
                    self.info['x1success'] += 1
                    self.model.center = self.samp.Y[x1idx]
                else:
                    self._success = 0

                # Update the trust region radius. 
                self.model.update_delta(rho, stepSize2delta, self.options)
                
            else: 
                # a model step
                rho = np.nan

                if self.samp.fY[x1idx] == np.min(self.samp.fY):
                    # Decide whether to move the iterate to x1. 
                    self._success = 2
                    self.info['x2success'] += 1
                    self.model.center = self.samp.Y[x1idx]
                elif self.samp.fY[x2idx] == np.min(self.samp.fY):
                    # Decide whether to move the iterate to x2. 
                    self._success = 2
                    self.info['x2success'] += 1
                    self.model.center = self.samp.Y[x2idx]
                else: 
                    self._success = 0
                
                # If the sample set is poised, then reduce TR radius. 
                if self.samp.m == self.options['sample_max']:
                    self.model.delta *= self.options['tr_shrink'] 

            # print iteration report
            if self.options['verbosity'] >= 2:
                print("| {:5d} | {} | {:11.5e} | {:11.5e} | {:9.6f} | {:9.6f} | {} |" \
                    .format(self.info['iteration'],
                            self._success,
                            self.samp.fY[x1idx],
                            self.samp.fY[x2idx],
                            self.model.delta, 
                            rho, 
                            self.samp.m
                            ))

            # Remove points that are too far away from the current TR. 
            self.samp.auto_delete(self.model, self.options)
        
        # end of one iteration
        self.info['iteration'] += 1

        # build an approximation model
        self.model.fit(self.samp)
        mY = np.array([self.model(y) for y in self.samp.Y])

        # Solve the trust region subproblem. 
        self.x1, m1 = self.model.minimize()
        self.info['predicted_decrease'] = self.model.c - m1 

        # x1
        if (self.info['predicted_decrease'] > 0) and (self.samp.distance(self.x1).min() >= 0.1 * self.model.delta):
            # If the new point is sufficiently far away from other points, 
            # add it to the sample set. 
            todelete = self.samp.addpoint(self.x1, copy.copy(self.model), self.options)
            # Keep the sample size under the capacity. 
            if todelete is not None:
                self.samp.rmpoint(todelete)
        else:
            # If the predicted decrease is not positive, increase the search radius
            # and change the new point to a poisedness improvement one. 
            # If the new point is too close to one of the sample points,
            # change it to a poisedness improvement point. 
            if (self.info['predicted_decrease'] <= 0) and (self.samp.m == self.options['sample_max']):
                self.model.delta *= 2
            
            self.x1, todelete = self.samp.poise(copy.copy(self.model))
            if todelete is not None:
                self.samp.rmpoint(todelete)
            self.samp.addpoint(self.x1, copy.copy(self.model), self.options)

            self.info['predicted_decrease'] = np.nan

        

        # Add a poisedness improvement point and delete an old one if necessary. 
        self.x2, todelete = self.samp.poise(copy.copy(self.model),\
             keep=[np.all(self.samp.Y == self.x1, axis=1).argmax()])
        if todelete is not None:
            self.samp.rmpoint(todelete)
        self.samp.addpoint(self.x2, copy.copy(self.model), self.options)

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
        # elif self.info['predicted_decrease'] <= self.options['stop_predict']:
        #     STOP = True
        #     print('Exiting because the minimum predicted decrease is reached.')
        
        if STOP and self.options['verbosity'] >= 1:
            print('***************** FINAL REPORT ************************')
            self.info['end_time'] = time.time()
            print('total elapsed time: {} seconds\n'.format(self.info['end_time'] - self.info['start_time']))
            print("|#iter|#x1success|#x2success|#fevals| best fvalue |final tr_radius|final predicted decrease|")
            print("|{:5d}|  {:5d}   |  {:5d}   | {:5d} | {:11.5e} |   {:9.6f}   |       {:11.5e}      |\n"
                .format(self.info['iteration'],
                        self.info['x1success'],
                        self.info['x2success'],
                        self.info['nfeval'], 
                        min(self.samp.fY), 
                        self.model.delta,
                        self.info['predicted_decrease']
                        ))

        return STOP



    @classmethod
    def optimize(cls, obj, x0, bounds=None, options=None):
        optimizer = dfo_tr(x0, bounds, options)

        while True:
            x = optimizer.ask()
            fx = [obj(x[0])]
            optimizer.tell(x,fx)
            if optimizer._stop():
                break
        idx = np.nanargmin(optimizer.samp.fY)
        return optimizer.samp.Y[idx], optimizer.samp.fY[idx], optimizer.info