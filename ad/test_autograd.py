from autograd import numpy as np
from autograd import jacobian

def init_array(arr):
    return np.array(arr)

class Constitutive:
    def __init__(self, g=1.4):
        self.gamma = g

    def eval_primitive(self, state):
        rho = state[0]
        u = state[1] / rho
        p = (self.gamma - 1) * (state[2] - 0.5 * rho * u * u)
        return rho, u, p

    def eval_speed_sound(self, rho, p):
        return np.sqrt(self.gamma * p / rho)

    def compute_flux(self, state):
        # Variables
        rho, u, p = self.eval_primitive(state)
        # FLux
        f = np.array([rho * u,
                      rho * u * u + p,
                      (state[2] + p) * u])
        return f

class LaxFried:
    def __init__(self, fluid):
        self.fluid = fluid
        self.compute_jacobian0 = jacobian(self.compute_residual, 0)
        self.compute_jacobian1 = jacobian(self.compute_residual, 1)

    def compute_residual(self, state0, state1, n, l):
        # Vars
        s = 0.5 * (state0 + state1)
        rho, u, p = self.fluid.eval_primitive(s)
        c = self.fluid.eval_speed_sound(rho, p)
        # Flux
        f = self.fluid.compute_flux(s)
        # Wave speed
        a = u * n + c
        # LF flux
        return (f * n - 0.5 * a * (state0 - state1)) * l

    def compute_jacobian(self, state0, state1, n, l):
        return (self.compute_jacobian0(state0, state1, n, l), self.compute_jacobian1(state0, state1, n, l))
