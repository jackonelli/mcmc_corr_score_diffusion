from functools import partial
from jax import jvp
import jax
import jax.numpy as jnp


x1 = jnp.array([1., 1.])
x2 = x1 + jnp.array([-0.2, 0.2])
diff = x2 - x1
"""
Q = jnp.array([[2., 1.], [1, 3]])

def f(x):
    return 0.5*jnp.dot(jnp.dot(x.T, Q), x)


def grad_f(x):
    return jnp.dot(x.T, Q)
"""


def grad_f(x):
    return jnp.array([jnp.exp(2*x[0]) + x[1], jnp.exp(x[1] + x[0])])


def func(s):
    return jnp.sum(grad_f(x1 + s*(x2 - x1)) * diff)


def tmp_f(s):
    return grad_f(x1 + s*(x2 - x1))


func_v = jax.vmap(func)


def taylor_exp():
    return jnp.sum((grad_f(x1) + 0.5*jvp(grad_f, (x1,), (diff,))[1]) * diff)


def trapz(n_points=50):
    s = jnp.linspace(0, 1, n_points)
    y = func_v(s)
    return jnp.trapz(y, s)


print('Taylor-expansion:', taylor_exp())
print('Trapz-500:', trapz(500))
print('Trapz-50:', trapz())
print('Trapz-2:', trapz(4))
#print('Answer:', f(x2)-f(x1))