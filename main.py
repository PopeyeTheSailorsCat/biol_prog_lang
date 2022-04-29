import numpy as np
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import parse_expr
import config
import optimization
import _logging
import argparse

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


def parse_input(file_path):
    with open(file_path, 'r') as file_input:
        expr_line = file_input.readline()
        _logging.info_message('\n \n' + ' *' * 10)
        _logging.info_message(f"Read input expression from {file_path}:{expr_line}")
        try:
            initial_values = np.array(file_input.readline().split(), dtype=float)
        except Exception as ex:
            _logging.error_message("Error during input initial reading: " + str(ex))

        line = file_input.readline()
        lower_bounds = []
        upper_bounds = []
        vars = []
        while line:
            var, lower_bound, upper_bound = line.split()
            vars.append(var)
            lower_bounds.append(float(lower_bound))
            upper_bounds.append(float(upper_bound))
            line = file_input.readline()

        if len(vars) != len(initial_values):
            _logging.error_message("Initial value dont have same size as variables")

        _logging.info_message(f"Function have {len(vars)} variables with bounds:")
        _dump = [_logging.info_message(f"{var}: ({low} , {up} ) ") for var, low, up in
                 zip(vars, lower_bounds, upper_bounds)]
        _logging.info_message(f"Starting position: {dict(zip(vars, initial_values))}")
        return expr_line, vars, initial_values, np.array([lower_bounds, upper_bounds])


@np_cache
def function_call(value_vector):
    vars_val = dict(zip(function_call.vars, value_vector))
    return parse_expr(function_call.expr, local_dict=vars_val)


def example_run(example_number, args):
    example_1_res = "this function have minimum of 0 in (3,2);(-2.805118, 3.131312),(-3.779310,-3.283186),((3.584428," \
                    "-1.848126)) "
    example_2_res = "this function have minimum of 0 in (0,0)"

    example_3_res = "this function have minimum of 0 in (0,0)"
    if example_number == 1:
        path = "examples\\example_1.txt"
        info = example_1_res
    elif example_number == 2:
        path = "examples\\example_2.txt"
        info = example_2_res
    else:
        path = "examples\\example_3.txt"
        info = example_3_res
    pos, val = main(path, args)
    _logging.info_message(info)


def main(path, args):
    try:
        expr_str, vars, initial, bounds = parse_input(path)
    except Exception as ex:
        _logging.error_message("Error during input reading")
        return -1
    function_call.expr = expr_str
    function_call.vars = vars
    try:
        opt_position, steps = optimization.optimize(function_call, initial, bounds, from_check_point=args.checkpoint,
                                                    check_point_path=args.checkpoint_path)
    except Exception as ex:
        _logging.error_message(f"Fatal error during optimization:{ex}")
        return -1

    try:
        _logging.write_results(vars, opt_position, function_call(opt_position), args.output)
    except Exception as ex:
        _logging.error_message(f"Fatal error during answer writing. Answer:{opt_position}, error:{ex}")
        return -1
    return opt_position, function_call(opt_position)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--example', dest='example',
                        help="Run example(1,2,3)")

    parser.add_argument('-p', '--path', dest='path',
                        help="give path to destination")

    parser.add_argument('-c', '--checkpoint', dest='checkpoint', action='store_true',
                        help="start from last checkpoint")

    parser.add_argument('-ch_pth', '--checkpoint_path', dest='checkpoint_path',
                        help="path to checkpoint from which we start input")

    parser.add_argument('-o', '--output', dest='output',
                        help="path to file with results")

    checkpoint = False
    args = parser.parse_args()
    if args.example:
        example_run(int(args.example), args)
    else:

        if not args.path:
            _logging.info_message(f"No path given, running {config.DEFAULT_EXAMPLE_PATH}")
            path = config.DEFAULT_EXAMPLE_PATH
        else:
            path = args.path

        if args.checkpoint:
            _logging.info_message(f"Starting input file from checkpoint")
        if args.checkpoint_path:
            _logging.info_message(f" As checkpoint we will use {args.checkpoint_path}")

        main(path, args)
