import config
import logging
import numpy as np


def info_message(msg):
    logging.info(msg)


def warning_message(msg):
    logging.warning(msg)


def error_message(msg):
    logging.error(msg)
    raise RuntimeError


def checkpoint(condition):
    with open(config.CHECKPOINT_FILE, 'w') as output_file:
        for elem in condition:
            if type(elem) is np.ndarray:
                output_file.write(" ".join(elem.astype(str)) + "\n")
                continue
            output_file.write(str(elem) + "\n")


def start_from_checkpoint(path):
    if path:
        open_file = path
    else:
        open_file = config.CHECKPOINT_FILE
    with open(open_file, 'r') as input_file:
        position = np.array(input_file.readline().split(" "), dtype=float)
        temp = float(input_file.readline())
        iter = float(input_file.readline())
    return [position, temp, iter]


def write_results(vars, result, fun_value, output_file):
    if output_file:
        out_name = output_file
    else:
        out_name = config.RESULT_FILE
    info_message(f"Optimal function value: {fun_value} in position:")
    with open(out_name, 'w') as output_file:
        for var, val in zip(vars, result):
            info_message(f"{var} = {val}")
            output_file.write(f"{var} = {val}")


logging.basicConfig(
    handlers=[
        logging.FileHandler('logging.log'),
        logging.StreamHandler()
    ]
    , encoding='utf-8', level=logging.INFO)
