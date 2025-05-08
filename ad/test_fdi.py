import numpy as np

def init_array(arr):
    return np.array(arr)

class Constitutive:
    def __init__(self, g=1.4):
        self.gamma = g
        self.eps = 1e-6

    def eval_primitive(self, state):
        rho = state[0]
        u = state[1] / rho
        p = (self.gamma - 1) * (state[2] - 0.5 * rho * u * u)
        return rho, u, p

    def eval_primitive_grad(self, state):
        vars = np.array(self.eval_primitive(state))
        grad = np.zeros((3, 3), dtype=float)
        for i_var in range(3):
            s = state[i_var]
            state[i_var] += self.eps
            grad[:, i_var] = (np.array(self.eval_primitive(state)) - vars) / self.eps
            state[i_var] = s
        return grad

    def eval_speed_sound(self, rho, p):
        return np.sqrt(self.gamma * p / rho)

    def eval_speed_sound_grad(self, rho, p):
        vars = self.eval_speed_sound(rho, p)
        grad = np.zeros((1, 2), dtype=float)
        rhop = rho + self.eps
        grad[:, 0] = (self.eval_speed_sound(rhop, p) - vars) / self.eps
        pp = p + self.eps
        grad[:, 1] = (self.eval_speed_sound(rho, pp) - vars) / self.eps
        return grad

    def compute_flux(self, state):
        rho, u, p = self.eval_primitive(state)
        return self._compute_flux(rho, u, p, state[2])

    def compute_flux_grad(self, state):
        rho, u, p = self.eval_primitive(state)
        dp_ds = self.eval_primitive_grad(state)
        df_dps = self._compute_flux_grad(rho, u, p, state[2])
        grad = np.zeros((3, 3), dtype=float)
        grad = df_dps[:, :3] @ dp_ds
        grad[:, 2] += df_dps[:, 3]
        return grad

    def _compute_flux(self, rho, u, p, e):
        f = np.zeros(3, dtype=float)
        f[0] = rho * u
        f[1] = rho * u * u + p
        f[2] = (e + p) * u
        return f

    def _compute_flux_grad(self, rho, u, p, e):
        vars = self._compute_flux(rho, u, p, e)
        grad = np.zeros((3, 4), dtype=float)
        rhop = rho + self.eps
        grad[:, 0] = (self._compute_flux(rhop, u, p, e) - vars) / self.eps
        up = u + self.eps
        grad[:, 1] = (self._compute_flux(rho, up, p, e) - vars) / self.eps
        pp = p + self.eps
        grad[:, 2] = (self._compute_flux(rho, u, pp, e) - vars) / self.eps
        ep = e + self.eps
        grad[:, 3] = (self._compute_flux(rho, u, p, ep) - vars) / self.eps
        return grad

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
        # Vars
        s = 0.5 * (state0 + state1)
        rho, u, p = self.fluid.eval_primitive(s)
        c = self.fluid.eval_speed_sound(rho, p)
        a = u * n + c
        # Jac
        idty = np.identity(3, dtype=float)
        dp_ds = self.fluid.eval_primitive_grad(s)
        dc_dp = self.fluid.eval_speed_sound_grad(rho, p)
        da_ds = np.array([dc_dp[0,0], n, dc_dp[0,1]]) @ dp_ds @ idty * 0.5
        jac0 = 0.5 * l * n * self.fluid.compute_flux_grad(s) @ idty
        jac0 -= 0.5 * l * np.stack((da_ds, da_ds, da_ds)) * np.diag(state0 - state1)
        jac0 -= 0.5 * l * a * idty
        jac1 = 0.5 * l * n * self.fluid.compute_flux_grad(s) @ idty
        jac1 -= 0.5 * l * np.stack((da_ds, da_ds, da_ds)) * np.diag(state0 - state1)
        jac1 += 0.5 * l * a * idty
        return jac0, jac1
