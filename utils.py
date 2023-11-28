from tensor import Tensor
import numpy as np
from tqdm import trange
import requests
import gzip
import os
import hashlib
import numpy as np

def sparse_categorical_crossentropy(out, Y):
  num_classes = out.shape[-1]
  YY = Y.flatten()
  y = np.zeros((YY.shape[0], num_classes), np.float32)
  # correct loss for NLL, torch NLL loss returns one per row
  y[range(y.shape[0]),YY] = -1.0*num_classes
  y = y.reshape(list(Y.shape)+[num_classes])
  y = Tensor(y)
  return out.mul(y).mean()

def NLLLoss(out, Y):
    num_classes = out.shape[-1]
    y = np.zeros((len(Y), num_classes), np.float32)
    y[range(y.shape[0]), Y] = -10.0
    y = Tensor(y)
    return out.mul(y).mean()

def uniform_init(n,m):
  ret = np.random.uniform(-1., 1., size=(n,m))/np.sqrt(n*m)
  return ret.astype(np.float32)




def training_loop(model, X_train, Y_train, optim, steps, 
                  lossfn=NLLLoss,  BS=128, transform = lambda x : x, target_transfrom = lambda x : x):
    losses, accuracies = [], []
    for i in (t := trange(steps)): 
        samp = np.random.randint(0, X_train.shape[0], size=(BS))
        x = Tensor(X_train[samp].reshape((-1, 28*28)).astype(np.float32))
        Y = Y_train[samp]

        out = model.forward(x)

        loss = lossfn(out, Y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        cat = np.argmax(loss.data, axis=1)
        accuracy = (cat == Y).mean()

        loss = loss.data
        losses.append(loss)
        accuracies.append(accuracy)
        t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))
    return [losses, accuracies]

