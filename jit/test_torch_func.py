import torch

def init_array(arr):
    return torch.tensor(arr)

@torch.compile
def eval_primitive(state, gamma):
    rho = state[0]
    u = state[1] / rho
    p = (gamma - 1) * (state[2] - 0.5 * rho * u * u)
    return rho, u, p

@torch.compile
def eval_speed_sound(rho, p, gamma):
    return torch.sqrt(gamma * p / rho)

@torch.compile
def compute_flux(state, gamma):
    # Variables
    rho, u, p = eval_primitive(state, gamma)
    c = eval_speed_sound(rho, p, gamma)
    # FLux
    f = torch.zeros(3, dtype=float)
    f[0] = rho * u
    f[1] = rho * u * u + p
    f[2] = (state[2] + p) * u
    return f

@torch.compile
def compute_residual(state0, state1, n, l, gamma):
    # Vars
    s = 0.5 * (state0 + state1)
    rho, u, p = eval_primitive(s, gamma)
    c = eval_speed_sound(rho, p, gamma)
    # Flux
    f = compute_flux(s, gamma)
    # Wave speed
    a = u * n + c
    # LF flux
    return (f * n - 0.5 * a * (state0 - state1)) * l
