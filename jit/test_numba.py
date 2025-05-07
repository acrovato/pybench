import numba as nb
import numpy as np

def init_array(arr):
    return np.array(arr)

class Constitutive:
    def __init__(self, g=1.4):
        self.gamma = g

    @staticmethod
    @nb.njit
    def _eval_primitive(gamma, state):
        rho = state[0]
        u = state[1] / rho
        p = (gamma - 1) * (state[2] - 0.5 * rho * u * u)
        return rho, u, p

    def eval_primitive(self, state):
        return self._eval_primitive(self.gamma, state)

    @staticmethod
    @nb.njit
    def _eval_speed_sound(gamma, rho, p):
        return np.sqrt(gamma * p / rho)

    def eval_speed_sound(self, rho, p):
        return self._eval_speed_sound(self.gamma, rho, p)

    @staticmethod
    @nb.njit
    def _compute_flux(eval_primitive, eval_speed_sound, gamma, state):
        # Variables
        rho, u, p = eval_primitive(gamma, state)
        c = eval_speed_sound(gamma, rho, p)
        # FLux
        f = np.zeros(3, dtype=float)
        f[0] = rho * u
        f[1] = rho * u * u + p
        f[2] = (state[2] + p) * u
        return f

    def compute_flux(self, state):
        return self._compute_flux(self._eval_primitive, self._eval_speed_sound, self.gamma, state)

class LaxFried:
    def __init__(self, fluid):
        self.fluid = fluid

    @staticmethod
    #@nb.njit # TODO does not work because fluid is not a numba type => solution is to pass required static methods directly
    def _compute_residual(fluid, state0, state1, n, l):
        # Vars
        s = 0.5 * (state0 + state1)
        rho, u, p = fluid.eval_primitive(s)
        c = fluid.eval_speed_sound(rho, p)
        # Flux
        f = fluid.compute_flux(s)
        # Wave speed
        a = u * n + c
        # LF flux
        return (f * n - 0.5 * a * (state0 - state1)) * l

    def compute_residual(self, state0, state1, n, l):
        return self._compute_residual(self.fluid, state0, state1, n, l)
