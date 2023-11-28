from tensor import Tensor
import numpy as np
from datasets import fetch_mnist
from tqdm import trange
from optim import Adam, SGD, Adadelta, RMSProp
from utils import training_loop

def layer_init_uniform(m,n):
  ret = np.random.uniform(-1., 1., size=(m,n))/np.sqrt(m*n)
  return ret.astype(np.float32)

X_train, Y_train, X_test, Y_test = fetch_mnist()

# create a model
class TinyBobNet:
  def __init__(self):
    self.l1 = Tensor(layer_init_uniform(784, 128))
    self.l2 = Tensor(layer_init_uniform(128, 10))

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

model = TinyBobNet()
#optim = SGD([model.l1, model.l2], lr=0.001)
#optim = Adam([model.l1, model.l2], lr=0.001)
#optim = Adadelta([model.l1, model.l2], lr = 0.001)
optim = RMSProp([model.l1, model.l2], lr = 0.001)

BS = 128
losses, accuracies = [], []
for i in (t := trange(1000)):
  samp = np.random.randint(0, X_train.shape[0], size=(BS))
  
  x = Tensor(X_train[samp])
  Y = Y_train[samp]
  y = np.zeros((len(samp),10), np.float32)
  # correct loss for NLL, torch NLL loss returns one per row
  y[range(y.shape[0]),Y] = -10.0
  y = Tensor(y)
  
  # network
  out = model.forward(x)

  # NLL loss function
  loss = out.mul(y).mean()
  loss.backward()
  optim.step()
  
  cat = np.argmax(out.data, axis=1)
  accuracy = (cat == Y).mean()
  
  # printing
  loss = loss.data
  losses.append(loss)
  accuracies.append(accuracy)
  t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

# evaluate
def numpy_eval():
  Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28))))
  Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
  return (Y_test == Y_test_preds).mean()

accuracy = numpy_eval()
print("test set accuracy is %f" % accuracy)
assert accuracy > 0.95