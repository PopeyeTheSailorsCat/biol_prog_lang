import math
import time
from sympy.parsing.sympy_parser import parse_expr
import optimization
import _logging
import numpy as np
from sympy import Float

f_1 = "x+x"
f_2 = "x/2"
f_3 = "x*x"
f_4 = "sqrt(x)"
f_5 = "log(x)"
f_6 = "exp(x)"
f_7 = "x/(x+2)"
func = [f_1, f_2, f_3, f_4, f_5, f_6, f_7]

from functools import lru_cache, wraps


def np_cache(function):
    @lru_cache()
    def cached_wrapper(hashable_array):
        array = np.array(hashable_array)
        return function(array)

    @wraps(function)
    def wrapper(array):
        return cached_wrapper(tuple(array))

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


@np_cache
def function_call(value_vector):
    vars_val = dict(zip(function_call.vars, value_vector))
    return Float(parse_expr(function_call.expr, local_dict=vars_val))


def calc_time_complexity():
    start = time.time()
    for i in range(100):
        x = 0.55 + i
        for fun in func:
            x = x + x
            x = x / 2
            x = x * x
            x = np.sqrt(x)
            x = np.log(x)
            x = np.exp(x)
            x = x / (x + 2)
    end = time.time()
    T_0 = end - start
    print(f"T0:{T_0}")
    start = time.time()
    for i in range(100):
        function_call.expr = "x**2 + y**2 - x*y + x - y"
        function_call.vars = ["x", "y"]
        initial = np.array([0.55 + 0.2 * i, 0.55 + 0.2 * i], dtype=float)
        function_call(initial)
    end = time.time()
    T_1 = end - start
    print(f"T1:{T_1}")

    t_s = np.zeros(5)
    for i in range(5):
        print(f"iter :{i}")
        start = time.time()
        for j in range(100):
            function_call.expr = "x**2 + y**2 - x*y + x - y"
            function_call.vars = ["x", "y"]
            initial = np.array([-5, 5], dtype=float)
            bounds = np.array([[-10, -10], [10, 10]], dtype=float)
            opt_position, steps = optimization.optimize(function_call, initial, bounds)
        end = time.time()
        t_s[i] = end - start
    T_2 = np.mean(t_s)
    print(f"T2{T_2}")

    print(f"Time complexity: {(T_2 - T_1) / T_0}")


def accuracy():
    max_call = 100
    function_call.expr = "(x_1 ** 2 + x_2-11)**2 +(x_1 + x_2**2 - 7)**2"
    function_call.vars = ["x_1", "x_2"]
    initial = np.array([0, 0], dtype=float)
    bounds = np.array([[-5, -5], [5, 5]], dtype=float)
    opt_val = 0
    bests = np.zeros(20)
    for j in range(20):
        best = 10000
        print(j)
        for i in range(10, 100+1, 10):
            this_step_call = int(max_call * i / 100)
            opt_res, steps = optimization.optimize(function_call, initial, bounds, break_after_calling=this_step_call)
            if best > abs(function_call(opt_res) - opt_val):
                best = abs(function_call(opt_res) - opt_val)
        bests[j] = best

    print("worst:", np.max(bests))
    print("best:", np.min(bests))
    print("mean:", np.mean(bests))
    print("median:", np.median(bests))
    print("std:", np.std(bests))


calc_time_complexity()
accuracy()
