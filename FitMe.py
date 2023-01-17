import numpy as np
from hyperopt import hp, tpe, fmin


class FitMe():
    
    def __init__(self, dp3, ds3, data_P, data_S):
        data_part=10
        self.data_P = data_P[::data_part]
        self.data_S = data_S[::data_part]
        self.EmList = [1100,1300] # make it optional!
        self.dp3 = dp3
        self.ds3 = ds3

    def error(self, args):
        time_p = self.data_P["time"]
        time_s = self.data_S["time"]
        comulativeError = 0
        for Em in self.EmList:
            p = self.data_P["{}".format(Em)]
            s = self.data_S["{}".format(Em)]
            comulativeError += np.mean(np.sqrt((p/1e2-self.dp3(Em,time_p,*args))**2))/(max(p)-min(p))*1e2+\
                                np.mean(np.sqrt((s-self.ds3(Em,time_s,*args))**2))/(max(s)-min(s))
        return comulativeError

    def fitMe(self):
        s=np.array([1.94,2.025,2.2,2.22,2.14,2.22, 0.21,0.88,1.67,2.95,0.9,2.95, 0.79,1.63, 0.77,1.49,0.61])
        the_range=np.array([0.0001,0.0001,0.0001,0.0001, 0.0001,0.0001,  0.5,0.5,0.5,0.09, 0.5,0.09,  0.0001,0.0001, 0.5,0.5,0.5])
        s1=(1-0.5*the_range)*s
        s2=(1+0.2*the_range)*s
        space = [hp.uniform('e1',s1[0],s2[0]),hp.uniform('e11',s1[1],s2[1]),hp.uniform('e111',s1[2],s2[2]),hp.uniform('e12',s1[3],s2[3]),
                 hp.uniform('e2',s1[4],s2[4]),hp.uniform('e3',s1[5],s2[5]),
                 hp.uniform('b1',s1[6],s2[6]),hp.uniform('b11',s1[7],s2[7]),hp.uniform('b111',s1[8],s2[8]),hp.uniform('b12',s1[9],s2[9]),
                            hp.uniform('b2',s1[10],s2[10]),hp.uniform('b3',s1[11],s2[11]),
                            hp.uniform('smax',s1[12],s2[12]),hp.uniform('d33',s1[13],s2[13]),hp.uniform('a1',s1[14],s2[14]),hp.uniform('a2',s1[15],s2[15]),hp.uniform('a3',s1[16],s2[16])]
        b = fmin(self.error,space,algo=tpe.suggest,max_evals=35)
        return [b["e1"],b["e11"],b["e111"],b["e12"],b["e2"],b["e3"],b["b1"],b["b11"],b["b111"],b["b12"],b["b2"],b["b3"],b["smax"],b["d33"],b["a1"],b["a2"],b["a3"]]