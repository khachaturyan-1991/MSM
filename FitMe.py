import numpy as np
from hyperopt import hp, tpe, fmin

class FitMe():

    def __init__(self, dp3, ds3, data_P, data_S, EmList):
        data_part=10
        self.data_P = data_P[::data_part]
        self.data_S = data_S[::data_part]
        self.EmList = EmList
        self.dp3 = dp3
        self.ds3 = ds3
        self.parameters = {"e1": 1.9, "e11": 2.0, "e111": 2.2, "e12": 2.2, "e2": 2.1, "e3": 2.2, \
             "b1": 0.2, "b11": 0.9, "b111": 1.7, "b12": 2.9, "b2": 0.9, "b3": 2.9, \
             "Smax": 0.8, "d33": 1.6, "a1": 0.8, "a2": 1.5, "a3": 0.6}

    def error(self, args):
        for i, key in enumerate(self.argsList.keys()):
            self.parameters[key] = args[i]
        time_p = self.data_P["time"]
        time_s = self.data_S["time"]
        comulativeError = 0
        for Em in self.EmList:
            p = self.data_P["{}".format(Em)]
            s = self.data_S["{}".format(Em)]
            comulativeError += np.mean(np.sqrt((p-self.dp3(Em, time_p, self.parameters)*1e2)**2))/(max(p)-min(p))+\
                                np.mean(np.sqrt((s-self.ds3(Em, time_s, self.parameters))**2))/(max(s)-min(s))
        return comulativeError

    def fitMe(self, argsList, theRange=0.1, EvaluationSteps = 30):
        self.argsList = argsList
        space = [hp.uniform(key, (1-theRange)*argsList[key], (1+theRange)*argsList[key]) for key in argsList.keys()]
        fitted = fmin(self.error,space,algo=tpe.suggest,max_evals=EvaluationSteps)
        for key in fitted.keys():
            self.parameters[key] = fitted[key]
        return self.parameters