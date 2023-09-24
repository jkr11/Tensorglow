import numpy as np
from functools import partialmethod

class Ctx:
    def __init__(self, arg, *T):
        self.arg = arg
        self.parents = T
        self.saved = []
    
    def save_backward(self, *x):
        self.saved.extend(x)

class Tensor:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.cont = None
    
    def __str__(self):
        return "Tensor %r with grad %r" % (self.data, self.grad)
    
    def backward(self):
        if self.cont is None:
            return
        
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        assert(self.grad is not None)

        grads = self.cont.arg.backward(self.cont,self.grad)
        if len(self.cont.parents) == 1:
            grads = [grads]
        for t,g in zip(self.cont.parents, grads):
            t.grad = g
            t.backward()

class Func:
    def apply(self, arg, *x):
        ctx = Ctx(arg, self, *x)
        x = [self]+list(x)
        ret = Tensor(arg.forward(ctx, *[t.data for t in x]))
        ret.cont = ctx
        return ret
    
def reg(name, fn):
    setattr(Tensor, name, partialmethod(fn.apply, fn))

class Mul(Func):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_backward(input,weight)
        return input * weight
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved
        return input*grad_output, weight*grad_output
reg('mul', Mul)

class Sum(Func):
    @staticmethod
    def forward(ctx, input):
        ctx.save_backward(input)
        return np.array([input.sum()])
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved
        return grad_output * np.ones_like(input)
reg('sum', Sum)

class Dot(Func):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_backward(input)
        return input.dot(weight)
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved
        return grad_output.dot(weight.T), grad_output.T.dot(input).T
reg('dot', Dot)


class ReLU(Func):
    @staticmethod 
    def forward(ctx, input):
        ctx.save_backward(input)
        return np.maximum(input, 0)
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved
        grad_input = grad_output.copy()
        grad_input[input < 0] = 0
        return grad_input
reg('relu', ReLU)

class LogSoftmax(Func):
    # LS(x_i) = log (exp(x_i) / sum_i exp(x_i))
    @staticmethod
    def forward(ctx, input):
        #def exp_normalize(x):
        #    b = x.max(axis = 1)
        #    y = np.exp(x - b.reshape(-1))
        #    return y/y.sum()
        # https://stackoverflow.com/questions/61567597/how-is-log-softmax-implemented-to-compute-its-value-and-gradient-with-better
        c = input.max(axis = 1)
        output = input - c - np.log(np.exp(input - c.reshape(-1, 1).sum()))
        ctx.save_backward(output)
        return output
    
    @staticmethod
    def backward(ctx,grad_output):
        output, = ctx.saved
        return grad_output - np.exp(output)*grad_output.sum(axis=1).reshape((-1, 1))
reg('logsoftmax', LogSoftmax)

        
        
