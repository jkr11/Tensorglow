import numpy as np
from functools import partialmethod
from typing import Optional, Union


  

class Tensor:
  def __init__(self, data:Union[None, np.ndarray, int, float, bytes], requires_grad:Optional[bool]=None):
    #print(f"Using Tensor of type {type(data)} with {data}")
    self.data = data
    self.grad:Optional[Tensor] = None
    self.context:Optional[Function] = None
    self.requires_grad:Optional[bool] = None

  def __repr__(self):
    return "Tensor %r with grad %r" % (self.data, self.grad)

  def backward(self, allow_fill=True):
    #print("running backward on", self)
    if self.context is None:
      return

    if self.grad is None and allow_fill:
      assert self.data.size == 1 # 1 because of implicit gradient creation 
      self.grad = np.ones_like(self.data)

    # assert(self.grad is not None)

    grads = self.context.backward(self.context, self.grad)
    if len(self.context.parents) == 1:
      grads = [grads]
    for t,g in zip(self.context.parents, grads):
      if g.shape != t.data.shape:
        print("grad shape must match tensor shape in %r, %r != %r" %
          (self.context, g.shape, t.data.shape))
        assert(False)
      t.grad = g
      t.backward(False)

  @property
  def shape(self):
    return self.data.shape
  
  def numpy(self):
    return np.array(self.data)

  def mean(self):
    div = Tensor(np.array([1/self.data.size]))
    return self.sum().mul(div)
  
  def argmax(self, axis):
    return Tensor(np.argmax(self.data, axis=axis))
  
  def flatten(self):
    return self.data.reshape((-1, ))

class Function:
  def __init__(self, *tensors:Tensor):
    self.parents = tensors
    self.saved_tensors = []
    self.grads = [t.requires_grad for t in tensors]
    self.requires_grad = True if any(self.grads) else None if None in self.grads else False

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)

  # note that due to how partialmethod works, self and arg are switched
  def apply(self, arg, *x:Tensor) -> Tensor:
    ctx = arg(self, *x)
    ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
    if ctx.requires_grad: ret.context = ctx
    return ret

def register(name, fxn):
  setattr(Tensor, name, partialmethod(fxn.apply, fxn))


class Mul(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x*y

  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    return y*grad_output, x*grad_output
register('mul', Mul)

class Add(Function):
  @staticmethod
  def forward(ctx, x, y):
    return x+y

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, grad_output
register('add', Add)
    
class ReLU(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.maximum(input, 0)

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = grad_output.copy()
    grad_input[input < 0] = 0
    return grad_input
register('relu', ReLU)

"""
class GeLU(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    gelu = 0.5*input* (1 + np.tanh(np.sqrt(2/np.pi) * (input + 0.044715 * input**3)))
    return gelu
  @staticmethod
  def backward(ctx, grad_output):
"""   
    

class Dot(Function):
  @staticmethod
  def forward(ctx, input, weight):
    ctx.save_for_backward(input, weight)
    return input.dot(weight)

  @staticmethod
  def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    grad_input = grad_output.dot(weight.T)
    grad_weight = grad_output.T.dot(input).T
    return grad_input, grad_weight
register('dot', Dot)

class Sum(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.array([input.sum()])

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return grad_output * np.ones_like(input)
register('sum', Sum)

class Softmax(Function):
  @staticmethod
  def forward(ctx, input):
    m = input-input.max(axis=1)
    e = np.max(m)
    return e / e.sum(axis=1)

class LogSoftmax(Function):
  @staticmethod
  def forward(ctx, input):
    def logsumexp(x):
      #return np.log(np.exp(x).sum(axis=1))
      c = x.max(axis=1)
      return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))
    output = input - logsumexp(input).reshape((-1, 1))
    ctx.save_for_backward(output)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    output, = ctx.saved_tensors
    return grad_output - np.exp(output)*grad_output.sum(axis=1).reshape((-1, 1))
register('logsoftmax', LogSoftmax)
