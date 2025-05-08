import numpy as np

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
        f = np.zeros(3, dtype=float)
        f[0] = rho * u
        f[1] = rho * u * u + p
        f[2] = (state[2] + p) * u
        return f

class LaxFried:
    def __init__(self, fluid):
        self.fluid = fluid

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
        # Reference residual (should be readily available)
        res = self.compute_residual(state0, state1, n, l)
        # Jacobian
        eps = 1e-6
        jac0 = np.zeros((3, 3), dtype=float)
        jac1 = np.zeros((3, 3), dtype=float)
        for i_var in range(3):
            s0 = state0[i_var]
            state0[i_var] += eps
            jac0[:, i_var] = (self.compute_residual(state0, state1, n, l) - res) / eps
            state0[i_var] = s0
            s1 = state1[i_var]
            state1[i_var] += eps
            jac1[:, i_var] = (self.compute_residual(state0, state1, n, l) - res) / eps
            state1[i_var] = s1
        return jac0, jac1
