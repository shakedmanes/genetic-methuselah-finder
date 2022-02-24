class Settings:

    # Miscellaneous Settings
    BOARD_SIZE = 100
    ANIMATION_FILE_NAME = 'methuselah_found_X.gif'
    INITIAL_CONFIGURATION_FILE_NAME = 'methuselah_found_conf_X.png'

    # Genetic Algorithm Settings
    MAX_GENERATIONS = 50
    MAX_FITNESS = 1.0
    DIFF_SOLUTION_THRESHOLD = 0.3

    # Probabilites
    CROSSOVER_PROB = 0.6
    MUTATION_PROB = 0.5

    # Population and individual solution size
    POPULATION_SIZE = 10
    INDIVIDUAL_SIZE = 36  # Must be a exponent of number!

    # Fitness settings
    FITNESS_GENERATION_BOUND = 1550
