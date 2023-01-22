import numpy as np
import scipy.optimize
import scipy.integrate

class MSM_R():

    def __init__(self):
        self.tau0=0.7e-11 # should be optional?
        self.Ps=0.43#*(1+0.021*(Em*1e-3-0.833)/0.4) # where the nubmbers come from?
        self.epsabs = 1e-3

    def tau(self,E):
        return self.tau0*np.exp(E*1e4/self.Em)

    def L20(self,t, a):
        b2=a["b2"]
        b21=a["b111"]
        tau2=self.tau(a["e2"])
        tau21=self.tau(a["e111"])
        return b2/tau2*scipy.integrate.quad(lambda t1: (t1/tau2)**(b2-1)*np.exp(-(t1/tau2)**b2)*np.exp(-((t-t1)/tau21)**b21),0,t,epsabs=self.epsabs)[0]

    def L21(self,t, a):
        b2=a["b2"]
        tau2=self.tau(a["e2"])
        return 1-np.exp(-(t/tau2)**b2)-self.L20(t, a)

    def L101(self,t, a):
        b1=a["b1"]
        tau1=self.tau(a["e1"])
        b11=a["b11"]
        tau11=self.tau(a["e11"])
        return b1/tau1*scipy.integrate.quad(lambda t1: (t1/tau1)**(b1-1)*np.exp(-(t1/tau1)**b1)*np.exp(-((t-t1)/tau11)**b11),0,t,epsabs=self.epsabs)[0]

    def L102(self,t, a):
        b1=a["b1"]
        b12=a["b12"]
        tau1=self.tau(a["e1"])
        tau12=self.tau(a["e12"])
        return b1/tau1*scipy.integrate.quad(lambda t1: (t1/tau1)**(b1-1)*np.exp(-(t1/tau1)**b1)*np.exp(-((t-t1)/tau12)**b12),0,t,epsabs=self.epsabs)[0]

    def L12(self,t, a):
        b1=a["b1"]
        tau1=self.tau(a["e1"])
        return 1-np.exp(-(t/tau1)**b1)-self.L102(t, a)

    def core_t2_110(self,t,t1, a):
        b11=a["b11"]
        b111=a["b111"]
        tau11=self.tau(a["e11"])
        tau111=self.tau(a["e111"])
        return lambda t2: b11/tau11*((t2-t1)/tau11)**(b11-1)*np.exp(-((t2-t1)/tau11)**b11)*np.exp(-((t-t2)/tau111)**b111)

    def core_t1_110(self,t, a):
        b1=a["b1"]
        tau1=self.tau(a["e1"])
        return lambda t1: ((t1/tau1)**(b1-1))*np.exp(-(t1/tau1)**b1)*scipy.integrate.quad(self.core_t2_110(t,t1, a),t1,t,epsabs=self.epsabs)[0]

    def L110(self,t, a):
        b1=a["b1"]
        tau1=self.tau(a["e1"])
        return b1/tau1*scipy.integrate.quad(self.core_t1_110(t, a),0,t,epsabs=self.epsabs)[0]

    def core_t2_111(self,t,t1, a):
        b11=a["b11"]
        b111=a["b111"]
        tau11=self.tau(a["e11"])
        tau111=self.tau(a["e111"])
        return lambda t2: b11/tau11*((t2-t1)/tau11)**(b11-1)*np.exp(-((t2-t1)/tau11)**b11)*(1-np.exp(-((t-t2)/tau111)**b111))

    def core_t1_111(self,t, a):
        b1=a["b1"]
        tau1=self.tau(a["e1"])
        return lambda t1: ((t1/tau1)**(b1-1))*np.exp(-(t1/tau1)**b1)*scipy.integrate.quad(self.core_t2_111(t,t1, a),t1,t,epsabs=self.epsabs)[0]

    def L111(self,t, a):
        b1=a["b1"]
        tau1=self.tau(a["e1"])
        return b1/tau1*scipy.integrate.quad(self.core_t1_111(t, a),0,t,epsabs=self.epsabs)[0]

    def dP3(self, Em, t, a):
        eta11=np.sin(a["a1"])**2*np.sin(a["a2"])**2*np.sin(a["a3"])**2
        eta12=np.cos(a["a1"])**2*np.sin(a["a2"])**2*np.sin(a["a3"])**2
        eta2=np.cos(a["a2"])**2*np.sin(a["a3"])**2
        eta3=np.cos(a["a3"])**2
        self.Em = Em
        Ps=self.Ps
        b3=a["b2"]
        tau3=self.tau(a["e3"])
        return 2*Ps*(eta3*(1-np.exp(-(t/tau3)**b3))+eta2*(2/3*self.L20(t, a)+self.L21(t, a))+eta12*(1/3*self.L102(t, a)+self.L12(t, a))+eta11*(1/3*self.L101(t, a)+2/3*self.L110(t, a)+self.L111(t, a)))

    def dS3(self, Em, t, a):
        Ps=self.Ps
        eta11=np.sin(a["a1"])**2*np.sin(a["a2"])**2*np.sin(a["a3"])**2
        eta12=np.cos(a["a1"])**2*np.sin(a["a2"])**2*np.sin(a["a3"])**2
        eta2=np.cos(a["a2"])**2*np.sin(a["a3"])**2
        Smax=a["Smax"]
        d33=0.37
        Em=self.Em
        return -Smax*(eta2*self.L20(t, a)+eta12*self.L102(t, a)+eta11*(self.L101(t, a)+self.L110(t, a)))+d33*1e-7*Em*1e3*(self.dP3(Em, t, a)-Ps)+a["d33"]*1e-9*Em**2

    def tau_print(self, a):
        eta11=np.sin(a["a1"])**2*np.sin(a["a2"])**2*np.sin(a["a3"])**2
        eta12=np.cos(a["a1"])**2*np.sin(a["a2"])**2*np.sin(a["a3"])**2
        eta2=np.cos(a["a2"])**2*np.sin(a["a3"])**2
        eta3=np.cos(a["a3"])**2
        print (eta11,eta12,eta2,eta3,eta11+eta12+eta2+eta3)