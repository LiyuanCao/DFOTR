import numpy as np


class param:
    """the class of parameters for the DFO TR algorithm
    The input demension n is needed for initialization.
    """

    def __init__(self,n):
        if n > 1800:
            raise ValueError('This solver package does not handle problems with dimension larger than 1500 well. \n')

        self.n = n

        # the algorithm
        self.alg_model = 'quadratic' # the model used 
        self.alg_TRS = 'exact' # trust region subproblem
        self.alg_poised = False # poisedness improvement step

        # sample management parameters
        self.sample_initial = n+1          # initial sample size
        self.sample_min = int( (n+1)*1.1 ) # minimum sample size
        self.sample_max = min(3000, (n+1)*(n+2)/2)    # maximum sample size
        # a point is to be removed from the sample set, if it is
        # sample_toremove * tr_delta away from the current point
        self.sample_toremove = 100

        # trust region parameters
        self.tr_delta = 1.0  # initial delta (i.e. trust region radius)
        self.tr_toaccept = 0.0  # rho level to accept new point
        self.tr_toexpand = 0.5  # rho level to expand radius
        self.tr_toshrink = min(int(self.sample_min * 1.05), self.sample_max)
        # minimum number of sample points within (sample_toremove * tr_delta) to reduce radius
        self.tr_shrink = 0.95  # radius shrink factor
        self.tr_expand = 1.3  # radius expansion factor

        # stopping crieria parameters
        self.stop_delta = 1e-5
        self.stop_g = 1e-10  # threshold for norm of gradient
        self.stop_iter = 20000  # maximum number of iterations

        # verbosity
        self.verbosity = 2

    def overwrite(self, customizedPar):
        for key in customizedPar:
            if key not in self.__dict__.keys():
                raise ValueError("{!r} is not a valid parameter name. \n".format(key))
            self.__dict__[key] = customizedPar[key]

