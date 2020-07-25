from .option import param
import numpy as np

class GenericOptimizer:
    'a generic optimizer'

    def __init__(self, customOptions):
        self.options = {}

    def override_default_options(self, customOptions):
        for key in customOptions:
            if key not in self.options.keys():
                raise ValueError("{!r} is not a valid option name. \n".format(key))
            self.options[key] = customOptions[key]

    def optimize(self, objectiveFunction, startingPoint):
        self.n = len(startingPoint)
        self.log = {'numberOfFunctionEvaluation': 1}
        return startingPoint, objectiveFunction(startingPoint), self.log

    def ask(self, nAsk=1):
        listOfPointsToBeEvaluated = np.zeros((nAsk, self.n))
        return listOfPointsToBeEvaluated

    def tell(self, X, fX):
        pass