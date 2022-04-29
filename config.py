STARTING_TEMPERATURE = 6
STOPPING_TEMPERATURE = 0.01
ITERATION_PER_TEMPERATURE = 10

CLOSEST_STEP = 0  # possible random step using length of variable interval
FURTHER_STEP = 20

SAVING_DIVISION_EPSILON = 1e-3  # added to optimize division to avoid inf

# Cooling Schedule

COOLING_SCHEDULE = "geom"  # possible types : linear, geom, log_1, log_2
# log_1 is T = T/ln(K)
# log_2 is T = T/K

linear_param = 0.2  # T_new = T_old - linear_param
geom_param = 0.85  # T_new = T_old * geom_param

CHECKPOINT_FILE = "checkpoint.log"
CHECKPOINT_EVERY_COOLING = 100

RESULT_FILE = "result.txt"

DEFAULT_EXAMPLE_PATH = "examples\\example_1.txt"
