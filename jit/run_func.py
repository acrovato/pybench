import test_numpy_func
import test_numba_func
import test_jax_func
import test_torch_func
import time

def test(name, module):
    print(name)
    # Init values
    s0 = module.init_array([1.2, 170., 1e5])
    s1 = module.init_array([1.22, 175., 1.02e5])
    l = 0.1
    n = 1.0
    g = 1.4
    # Compile and run
    cpu = run(module, s0, s1, n, l, g)
    print('- comp time: {:.2f}'.format(cpu))
    cpu = run(module, s0, s1, n, l, g, 1e5)
    print('- exec time: {:.2f}'.format(cpu))

def run(module, s0, s1, n, l, g, n_run=1):
    # Run
    cpu = time.perf_counter()
    for i in range(int(n_run)):
        module.compute_residual(s0, s1, n, l, g)
    return time.perf_counter() - cpu

def main():
    test('NumPy', test_numpy_func)
    test('Numba', test_numba_func)
    test('JAX', test_jax_func)
    test('PyTorch', test_torch_func)

if __name__ == '__main__':
    main()
