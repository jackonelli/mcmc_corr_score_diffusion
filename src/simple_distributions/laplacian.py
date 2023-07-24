from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import jet
# jet.fact = lambda n: jax.lax.prod(range(1, n + 1))

def f(ws, wo, x):
    for w in ws:
        x = jax.lax.exp(x @ w)
    return jnp.reshape(x @ wo, ())

@jax.jit
@partial(jax.vmap, in_axes=(None, None, 0))
def laplacian_1(ws, wo, x):
    fun = partial(f, ws, wo)
    @jax.vmap
    def hvv(v):
        return jet.jet(fun, (x,), ((v, jnp.zeros_like(x)),))[1][1]
    return jnp.sum(hvv(jnp.eye(x.shape[0], dtype=x.dtype)))

@jax.jit
@partial(jax.vmap, in_axes=(None, None, 0))
def laplacian_2(ws, wo, x):
    fun = partial(f, ws, wo)
    in_tangents = jnp.eye(x.shape[0], dtype=x.dtype)
    pushfwd = partial(jax.jvp, jax.grad(fun), (x,))
    _, hessian = jax.vmap(pushfwd, out_axes=(None, 0))((in_tangents,))
    return jnp.trace(hessian)

@jax.jit
@partial(jax.vmap, in_axes=(None, None, 0))
def laplacian_3(ws, wo, x):
    fun = partial(f, ws, wo)
    return jnp.trace(jax.hessian(fun)(x))

def timer(f):
    from time import time
    f() # compile
    t = time()
    for _ in range(3):
        f()
    print((time() - t) / 3)

d = 256
ws = [jnp.zeros((d, d)) for _ in range(64)]
wo = jnp.zeros((d, 1))
x = jnp.zeros((512, d))

timer(lambda : jax.block_until_ready(laplacian_1(ws, wo, x)))
timer(lambda : jax.block_until_ready(laplacian_2(ws, wo, x)))
timer(lambda : jax.block_until_ready(laplacian_3(ws, wo, x)))

