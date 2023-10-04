from tensor import Tensor
import numpy as np
from datasets import fetch_mnist
from tqdm import trange
from optim import Adam, SGD
from utils import training_loop

X_train, Y_train, X_test, Y_test = fetch_mnist()

def Linear(m,n):
  ret = np.random.uniform(-1., 1., size=(m,n))/np.sqrt(m*n)
  return ret.astype(np.float32)

class Net:
  def __init__(self):
    self.l1 = Tensor(Linear(784, 128))
    self.l2 = Tensor(Linear(128, 10))

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()


model = Net()
optim = SGD([model.l1, model.l2], lr=0.01)
#optim = Adam([model.l1, model.l2], lr=0.01)

training_loop(model, X_train, Y_train, optim, 1000)


# evaluate
def numpy_eval():
  Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28))))
  Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
  return (Y_test == Y_test_preds).mean()

accuracy = numpy_eval()
print("test set accuracy is %f" % accuracy)
assert accuracy > 0.95