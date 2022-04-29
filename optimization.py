import numpy as np
import config
import random
from math import exp
import _logging
import sympy


def parameters_to_condition(position, temperature, iteration):
    _logging.checkpoint([position, temperature, iteration])


def condition_to_parameters(path):
    condition = _logging.start_from_checkpoint(path)
    position, temperature, iteration = condition
    scheduler = get_scheduler()
    if config.COOLING_SCHEDULE == "log_1":
        scheduler.count = iteration + 1
    elif config.COOLING_SCHEDULE == "log_2":
        scheduler.count = iteration
    return position, temperature, scheduler, iteration


def random_neighbour(position, borders):
    step = np.random.uniform(config.CLOSEST_STEP, config.FURTHER_STEP, size=len(position))
    step = np.where(np.random.random(len(step)) > 0.5, -step, step)
    position = position + step / 100 * (borders[1, :] - borders[0, :])
    sub_res = np.where(borders[0, :] < position, position, borders[0, :])
    return np.where(sub_res < borders[1, :], sub_res, borders[1, :])


def get_scheduler():
    if config.COOLING_SCHEDULE == "linear":
        if config.linear_param <= 0:
            _logging.error_message("Linear param <= than 0")
        return lambda x: x - config.linear_param
    elif config.COOLING_SCHEDULE == "geom":
        if config.geom_param >= 1:
            _logging.error_message("Geom param > 1")
        return lambda x: config.geom_param * x
    elif config.COOLING_SCHEDULE == "log_1":
        def log_1(x):
            log_1.count += 1
            return config.STARTING_TEMPERATURE / np.log(log_1.count)

        log_1.count = 1
        return log_1
    else:
        def log_2(x):
            log_2.count += 1
            return config.STARTING_TEMPERATURE / log_2.count

        log_2.count = 1
        return log_2


def optimize(func, start, borders, from_check_point=False, check_point_path=None):
    if from_check_point:
        try:
            position, T, scheduler, iteration_counter = condition_to_parameters(check_point_path)
            _logging.info_message(f"Start from checkpoint on {iteration_counter} iteration")
            _logging.info_message(f"position:{position}, T:{T}, scheduler: {config.COOLING_SCHEDULE}")
        except Exception as ex:
            _logging.error_message(f"ERROR during checkpoint reading:{ex}")
            return
    else:
        scheduler = get_scheduler()
        iteration_counter = 0
        T = config.STARTING_TEMPERATURE
        _logging.info_message(f"Using {config.COOLING_SCHEDULE} scheduler and starting t = {T}")
        position = start
    checkpoint_counter = 0
    log_steps = [position]
    while T > config.STOPPING_TEMPERATURE:
        iteration_counter += 1
        for i in range(config.ITERATION_PER_TEMPERATURE):
            neighbour = random_neighbour(position, borders)
            try:
                diff = func(neighbour) - func(position)
            except Exception as ex:
                _logging.warning_message("Error during function calculation:" + str(ex))
                return

            if diff.has(sympy.oo, -sympy.oo, sympy.zoo, sympy.nan):
                _logging.warning_message(f"Error during function calculation:{diff}")
                continue

            try:
                if diff < 0 or random.random() < exp(-diff / (T + config.SAVING_DIVISION_EPSILON)):
                    position = neighbour
                    log_steps.append(position)
            except Exception as ex:
                _logging.warning_message(f"Error during position changing:{ex}")
                continue

        T = scheduler(T)
        checkpoint_counter += 1
        if checkpoint_counter == config.CHECKPOINT_EVERY_COOLING:
            try:
                parameters_to_condition(position, T, iteration_counter)
            except Exception as ex:
                _logging.warning_message(f"Error during checkpoint saving:{ex}")
            checkpoint_counter = 0
            _logging.info_message(f"Checkpoint on iteration:{iteration_counter}")
    return position, log_steps
