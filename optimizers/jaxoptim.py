import optax

# Define initial parameters
params = jax.random.normal(key, (num_layers, num_units))


# Define Adam optimizer
adam_optimizer = optax.adam(learning_rate=1e-3)

# Define SGD optimizer
sgd_optimizer = optax.sgd(learning_rate=0.01)

# Optimization loop
for _ in range(num_iterations):
    # Compute gradients
    grads = jax.grad(loss)(params, ...)
    
    # Update parameters with Adam optimizer
    updates, optimizer_state = adam_optimizer.update(grads, optimizer_state, params)
    params = optax.apply_updates(params, updates)

    # Compute gradients again if needed
    grads = jax.grad(loss)(params, ...)

    # Update parameters with SGD optimizer
    updates, optimizer_state = sgd_optimizer.update(grads, optimizer_state, params)
    params = optax.apply_updates(params, updates)
