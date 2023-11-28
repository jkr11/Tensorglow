# optim/adam.py
from tensor import Tensor
import numpy as np




class Optimizer:
    def __init__(self, params):
        self.params = params

    def num(self, x):
        s = np.array(x)
        return Tensor(s)

    def zero_grad(self):
        for param in self.params : param.grad = None

class SGD(Optimizer):
    def __init__(self, params, lr=0.001):
        super(SGD, self).__init__(params)  
        self.lr = lr

    def step(self):
        for t in self.params:
            t.data -= self.lr * t.grad
            
    
    
class RMSProp(Optimizer):
    def __init__(self, params, lr=0.001,eps=1e-8,mu=0.9): # mu = decay, lr=0.001, decay = 0.9, eps = 1e-8
        super(RMSProp, self).__init__(params)
        self.lr = lr
        self.eps = eps
        self.mu = mu

        self.s = [np.zeros_like(t) for t in self.params]

    def step(self):
        for i,t in enumerate(self.params):
            self.s[i] = self.mu * self.s[i] + (1 - self.mu) * t.grad.data**2
            t.data -= (self.lr / (np.sqrt(self.s[i] + self.eps)))*t.grad.data


class Adadelta(Optimizer):
    def __init__(self, params, lr=0.001, rho=0.9, eps=1e-6, decay=0.1):
        super(Adadelta, self).__init__(params)
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.decay = decay
        self.t = 0

        self.v = [np.zeros_like(x.data) for x in self.params]
        self.u = [np.zeros_like(x.data) for x in self.params]

    
    def step(self):
        #self.t += 1
        for i,t in enumerate(self.params):
            if self.decay != 0:
                t.grad.data = t.grad.data + self.decay * t.data
            self.v[i] = self.v[i] * self.rho + np.square(t.grad.data)*(1 - self.rho)
            dx = (np.sqrt(self.u[i] + self.eps))/(np.sqrt(self.v[i] + self.eps)) * t.grad.data
            self.u[i] = self.u[i] * self.rho + np.square(dx) * (1 - self.rho)
            t.data = t.data - self.lr * dx



class Adam(Optimizer):
    def __init__(self, params, amsgrad=False, maximize=False, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        super(Adam, self).__init__(params)  
        
        self.maximize = maximize
        self.amsgrad = amsgrad
        

        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        #[self.num(x) for x in [lr, b1, b2, eps, 0, 1]]

        self.m = [np.zeros_like(t) for t in self.params]
        self.v = [np.zeros_like(t) for t in self.params]
        if amsgrad:
            self.vmax = [np.zeros_like(t) for t in self.params]

    

    def step(self): 
        
        # save calculating vhat and mhat & 2 in-loop multiplications
        #denom = self.lr * (np.sqrt(1 - np.power(self.b2, self.t)) / (1 - np.power(self.b1, self.t)))
        for i, t in enumerate(self.params):
            self.t = self.t + 1
            #if self.maximize:
            #    grad = -t.grad.data
            #else:
            #    grad = t.grad.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * t.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.square(t.grad)
            mhat = self.m[i] / (1. - self.b1**self.t)
            vhat = self.v[i] / (1. - self.b2**self.t)
            t.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
            #t.data -= denom * self.m[i] / (np.sqrt(self.v[i]) + self.eps)
        





