from jax import random
from jax.experimental import stax

key = random.PRNGKey(0)
_, init_params = stax.Dense(128)
_, apply_fn = stax.serial(
    stax.Dense(128),
    stax.Relu,
    stax.Dense(10)
)
params = init_params(key, (1, 28*28))

def loss(params, inputs, targets):
    predictions = apply_fn(params, inputs)
    return jnp.mean((predictions - targets)**2)

grad_loss = grad(loss)
