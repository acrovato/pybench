import test_fd
import test_fdi
import test_cs
import test_autograd
import test_jax
import time

def test(name, module):
    print(name)
    # Init values
    s0 = module.init_array([1.2, 170., 1e5])
    s1 = module.init_array([1.22, 175., 1.02e5])
    l = 0.1
    n = 1.0
    # Init objects
    fluid = module.Constitutive(1.4)
    flux = module.LaxFried(fluid)
    # Compile and run
    cpu = run(flux, s0, s1, n, l)
    print('- comp time: {:.2f}'.format(cpu))
    cpu = run(flux, s0, s1, n, l, 1e4)
    print('- exec time: {:.2f}'.format(cpu))

def run(flux, s0, s1, n, l, n_run=1):
    # Run
    cpu = time.perf_counter()
    for i in range(int(n_run)):
        flux.compute_jacobian(s0, s1, n, l)
    return time.perf_counter() - cpu

def main():
    test('FD', test_fd)
    test('FD (inner)', test_fdi)
    test('CS', test_cs)
    test('AutoGrad', test_autograd)
    test('JAX', test_jax)

if __name__ == '__main__':
    main()
