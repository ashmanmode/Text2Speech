import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np

x = T.dscalar()
y = T.exp(T.sin(x**2))

print type(y)

f = theano.function([x],y)
print f(2.3)

fp = T.grad(y,wrt=x)
fnew = theano.function([x],fp)

print fnew(2.3)

#Simple tensors

