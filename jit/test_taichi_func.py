import taichi as ti
import taichi.math as tm
import numpy as np

# TODO: this is probably not the best way to use Taichi...
ti.init(arch=ti.cpu)

def init_array(arr):
    return np.array(arr)

@ti.func
def eval_primitive(state, gamma):
    rho = state[0]
    u = state[1] / rho
    p = (gamma - 1) * (state[2] - 0.5 * rho * u * u)
    return rho, u, p

@ti.func
def eval_speed_sound(rho, p, gamma):
    return tm.sqrt(gamma * p / rho)

@ti.func
def compute_flux(state, gamma):
    # Variables
    rho, u, p = eval_primitive(state, gamma)
    c = eval_speed_sound(rho, p, gamma)
    # FLux
    f = ti.Vector([0., 0., 0.])
    f[0] = rho * u
    f[1] = rho * u * u + p
    f[2] = (state[2] + p) * u
    return f

@ti.kernel
def _compute_residual(lflux: ti.types.ndarray(), state0: ti.types.ndarray(), state1: ti.types.ndarray(), n: float, l: float, gamma: float):
    # Vars
    s = ti.Vector([0., 0., 0.])
    for i in range(3):
        s[i] = .5 * (state0[i] + state1[i])
    rho, u, p = eval_primitive(s, gamma)
    c = eval_speed_sound(rho, p, gamma)
    # Flux
    f = compute_flux(s, gamma)
    # Wave speed
    a = u * n + c
    # LF flux
    for i in range(3):
        lflux[i] = (f[i] * n - 0.5 * a * (state0[i] - state1[i])) * l

def compute_residual(state0, state1, n, l, gamma):
    flux = np.zeros(3)
    _compute_residual(flux, state0, state1, n, l, gamma)
    return flux
