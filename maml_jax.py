import time

import numpy.random as npr

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax
from utils.dataset_alt import *

def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
    return [(scale * rng.randn(m, n), scale * rng.randn(n)) 
            for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

def bernoulli_logpdf(logits, y):
    """Bernoulli log pdf of data x given logits."""
    return -jnp.sum(jnp.logaddexp(0., jnp.where(y, -1., 1.) * logits))

def predict(params, inputs):
    activations = inputs
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jnp.tanh(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return logits - logsumexp(logits, axis=1, keepdims=True)

def loss(params, data):
    inputs, targets = data
    logits_y = predict(params, inputs)
    return bernoulli_logpdf(logits_y, targets)



if __name__ == "__main__":
    rng = random.PRNGKey(0)
    
    layer_sizes = [51, 256, 256, 1]
    param_scale = 0.1
    step_size = 0.001
    num_epochs = 10
    
    amine_list, x_t, y_t, x_v, y_v, all_data, all_labels = process_dataset(train_size=10, active_learning_iter=10, 
                                                                           verbose=True, cross_validation=True, full=True,
                                                                           active_learning=True, w_hx=True, w_k=True)
    num_amines = len(amine_list)
    
    @jit
    def update(params, batch):
        grads = grad(loss)(params, batch)
        return [(w - step_size * dw, b - step_size * db) 
                for (w, b), (dw, db) in zip(params, grads)]
    
    @jit
    def meta_update(params, batch):
        return update(update(params, batch), batch)

    params = init_random_params(param_scale, layer_sizes)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        for amine in amine_list:
            params = meta_update(params,(x_t[amine],y_t[amine]))
        epoch_time = time.time() - start_time
        
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))