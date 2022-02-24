import math
import random
import matplotlib.pyplot as plt
import numpy as np

from seagull import Board, Simulator
from seagull.lifeforms import Custom
from seagull.rules import conway_classic
from loguru import logger
from warnings import filterwarnings

from settings import Settings


class GeneticMethuselahFinder:

    __fitness_generation_bound = Settings.FITNESS_GENERATION_BOUND
    __board_size = Settings.BOARD_SIZE
    __population_size = Settings.POPULATION_SIZE
    __individual_size = Settings.INDIVIDUAL_SIZE
    __crossover_prob = Settings.CROSSOVER_PROB
    __mutation_prob = Settings.MUTATION_PROB

    __default_crossover_point = __individual_size // 2

    __animation_file_name = Settings.ANIMATION_FILE_NAME
    __initial_configuration_file_name = Settings.INITIAL_CONFIGURATION_FILE_NAME

    @classmethod
    def run_genetic_algorithm(cls, max_generations=50, max_fitness=1, diff_solution_threshold=0.3):
        """
        Runs the genetic algorithm to find Methuselahs solutions, by predefined conditions.\n
        During the run of the genetic algorithm, print out information of the genetic algorithm process,
        including the solutions found and fitness value of each solution.\n
        When a solution found and match the stopping conditions,
        saving solution animation history and initial configuration of the solution in files.\n
        Also at the end, shows the solution in array representation and fitness and lifespan.\n

        :param max_generations: Max generation value to stop the genetic algorithm (Default 50).
        :param max_fitness: Max fitness value to stop the genetic algorithm (Default 1)
        :param diff_solution_threshold: The percentage of difference threshold allowed in the solution population.
        Any value below that will cause random mutation and random crossover in the next generation (Default 0.3).
        """

        # Conditions and configurations for the genetic algorithm
        generation_condition = lambda generation: generation < max_generations
        fitness_condition = lambda fitness: fitness < max_fitness
        current_generation = 0
        force_mutation_random = False
        crossover_point = cls.__default_crossover_point

        print('Initializing first solution population...')
        population = cls.initialize_population()
        print('Calculating the fitness of the solutions...')
        population_fitness = [cls.fitness(solution) for solution in population]
        current_max_fitness = max(population_fitness)

        # Combining conditions into one condition
        condition =\
            lambda values:\
            generation_condition(values['generation']) and\
            fitness_condition(values['fitness'])

        print(f'Current generation - {current_generation}, Fitness - {current_max_fitness}')
        print(f'Current population: \n{population}')
        print(f'Current fitness values: \n{population_fitness}')
        print('Starting genetic algorithm generations')

        while condition({
            'generation': current_generation,
            'fitness': current_max_fitness
        }):
            current_generation += 1
            new_population = []

            # Generating new population using selection, cross over and mutate.
            while len(new_population) != len(population):
                parent_one = cls.select(population_fitness)
                parent_two = cls.select(population_fitness)

                if parent_one == parent_two:
                    offspring = cls.mutate(population[parent_one], force_mutation_random)
                    new_population.append(offspring)
                else:
                    offspring = cls.crossover(population[parent_one], population[parent_two], crossover_point)
                    new_population.append(cls.mutate(offspring, force_mutation_random))

            population = new_population
            population_fitness = [cls.fitness(solution) for solution in population]
            current_max_fitness = max(population_fitness)
            force_mutation_random = False
            crossover_point = cls.__default_crossover_point

            print(f'Current generation - {current_generation}, Fitness - {current_max_fitness}')
            print(f'Current population: \n{population}')
            print(f'Current fitness values: \n{population_fitness}')

            # Check uniqueness of fitness values of solutions
            # (if below the threshold, need to force mutations to build new solutions)
            unique_fitness = np.unique(population_fitness, axis=0)
            if len(unique_fitness) / len(population_fitness) < diff_solution_threshold:
                print(f'Solution uniqueness is below the threshold {diff_solution_threshold}\n'
                      f'Forcing random mutation and crossover point next generation.')
                force_mutation_random = True
                crossover_point = np.random.randint(0, cls.__individual_size)

        print('Done running genetic algorithm, found maximum solution.')
        max_solution = population[population_fitness.index(current_max_fitness)]

        # Saving the solution initial configuration and animation history
        print('Saving solution animation and initial configuration...')
        solution_history = cls.save_solution(max_solution)

        print(f'Max Solution: {max_solution}\nFitness: {current_max_fitness}\n'
              f'Lifespan: {cls.__solution_lifespan(solution_history).get("lifespan")} Generations\n')

    @classmethod
    def initialize_population(cls):
        """
        Generates a population of unique individual solutions

        :return: List of np.array of unique solutions.
        """
        population = []
        existing_solutions = {}

        while len(population) != GeneticMethuselahFinder.__population_size:
            solution = cls.generate_solution()
            solution_str = np.array_str(solution)

            if existing_solutions.get(solution_str, None) is None:
                population.append(solution)
                existing_solutions[solution_str] = True

        return population

    @classmethod
    def generate_solution(cls):
        """
        Generates a random solution based on the individual solution size given to the genetic algorithm.

        :return: Array which represent solution to the genetic algorithm.
        """

        # Choose alive (ones) and dead (zeros) cells
        ones = random.randint(0, cls.__individual_size)
        zeros = cls.__individual_size - ones

        # Randomize and rearrange the living and dead cells
        solution = np.array([0] * zeros + [1] * ones)
        np.random.shuffle(solution)

        return solution

    @classmethod
    def fitness(cls, solution):
        """
        Calculates fitness value of a given solution.\n
        The fitness value is determined by the following rule:
        Solution Peak Cell Coverage + Lifespan Percentage of Solution.\n

        Solution Peak Cell Coverage - The cell coverage percentage based on the solution lifespan.\n
        Lifespan Percentage of Solution - The calculated non repeating lifespan of the solution divide the number
        of generations the simulated GOL board run with the solution.\n

        :param solution: Solution to calculate fitness for
        :return: Float fitness value (0 to 2).
        """

        # Creating a GOL board which includes the solution in the center of the board, as matrix.
        board = Board(size=(cls.__board_size, cls.__board_size))
        board.add(
            Custom(solution.reshape(int(math.sqrt(cls.__individual_size)), int(math.sqrt(cls.__individual_size)))),
            loc=(cls.__board_size // 2, cls.__board_size // 2)
        )

        # Simulate the solution steps in the GOL board.
        simulator = Simulator(board)
        simulator_stats = simulator.run(
            conway_classic,
            iters=cls.__fitness_generation_bound
        )

        simulated_history = simulator.get_history()

        # Calculate the lifespan percentage of the solution based on the generations the simulator run on.
        fitness_lifespan_value = \
            cls.__solution_lifespan(simulated_history).get('lifespan') / cls.__fitness_generation_bound

        return simulator_stats.get('peak_cell_coverage') + fitness_lifespan_value

    @classmethod
    def crossover(cls, parent_one, parent_two, crossover_point):
        """
        Crossover 2 parent solutions into offspring solution, based on the crossover probability
        of the genetic algorithm.\n
        The crossover takes a crossover point, and intersect the two parents at the point.\n
        After that, combining the intersection parts creates 2 new offspring, which
        only one of them is chosen (randomly).

        :param parent_one: Solution parent one
        :param parent_two: Solution parent two
        :param crossover_point: Crossover point to perform the crossover from
        :return: Offspring solution based on the 2 parent crossover, or one of the parents
        if the crossover probability not occurred.
        """
        selected_offspring = random.randint(0, 1)

        if cls.__probability(cls.__crossover_prob):
            # Copy both parent, avoiding destruction of parents
            copy_parent_one = parent_one.copy()
            copy_parent_two = parent_two.copy()

            # Intersect on the crossover point
            copy_parent_two[:crossover_point], copy_parent_one[:crossover_point] = \
                copy_parent_one[:crossover_point], parent_two[:crossover_point]

            if selected_offspring == 0:
                return copy_parent_one
            return copy_parent_two

        if selected_offspring == 0:
            return parent_one
        return parent_two

    @classmethod
    def mutate(cls, solution, force_random=False):
        """
        Mutate an existing solution, based on the mutation probability of the genetic algorithm.<br>
        The mutation performed is `Value Flipping` (1 to 0, 0 to 1).\n
        If `force_random` is True, forcing mutation and shuffle the solution values
        (Intentionally like value flipping, but more aggressive).

        :param solution: Solution to mutate.
        :param force_random: Boolean indicates if need to force the mutation
        and using random shuffle (Default is False).
        :return: Mutated solution, or the given solution if mutation probability not occurred
        or `force_random` is False.
        """
        if cls.__probability(cls.__mutation_prob) or force_random:
            if force_random:
                np.random.shuffle(solution)
            else:
                return 1 - solution
        return solution

    @staticmethod
    def select(population_fitness):
        """
        Selects a parent from the solution population for crossover operation.
        Using `Roulette Wheel Selection` mechanism.

        :param population_fitness: Array of population fitness of the solution population
        :return: The index of the solution which picked from the solution population,
        based on the population fitness.
        """
        sum_fitness = sum(population_fitness)
        selection_probs = [fitness / sum_fitness for fitness in population_fitness]
        return np.random.choice(len(population_fitness), p=selection_probs)

    @classmethod
    def save_solution(cls, solution):
        """
        Save the solution initial configuration and animation history in files.

        :param solution: Solution to save.
        :return: The solution history (array of board states of the solution as seagull library returns).
        """
        board = Board(size=(cls.__board_size, cls.__board_size))
        board.add(
            Custom(solution.reshape(int(math.sqrt(cls.__individual_size)), int(math.sqrt(cls.__individual_size)))),
            loc=(cls.__board_size // 2, cls.__board_size // 2)
        )
        initial_configuration_img, _ = board.view()
        initial_configuration_img.savefig(cls.__initial_configuration_file_name)
        plt.close(initial_configuration_img)

        simulator = Simulator(board)
        simulated_board = simulator.run(
            conway_classic,
            iters=cls.__fitness_generation_bound
        )

        animation = simulator.animate()
        animation.save(cls.__animation_file_name)

        return simulator.get_history()

    @staticmethod
    def __probability(probability):
        """
        Test the probability given.

        :param probability: Probability to test.
        :return: True if the probability occurs, otherwise False.
        """
        return random.random() < probability

    @staticmethod
    def __solution_lifespan(solution_history):
        """
        Calculates the solution lifespan, using the solution history on the GOL board.\n
        It finds the unique states in the board, and iterating along the unique array
        until finding the duplication of unique state.\n
        Because the GOL is has deterministic rules, the unique state will evolve to previous
        already seen state, and therefore, the lifespan can be stopped here.

        :param solution_history: np.array of board states, from the seagull library.
        :return: Dictionary which includes:<br>
        lifespan - Lifespan of the solution.<br>
        repetition - Size of the first noticed repetition in the lifespan of the solution.<br>
        start - The start index of the repetition in the solution history.<br>
        end - The end index of the repetition in the solution history.
        """
        _, inverse_indices = np.unique(solution_history, axis=0, return_inverse=True)

        # Creating unique map to save already seen unique states
        unique_map = dict()
        solution_history_length = len(inverse_indices)

        for index in range(solution_history_length):
            unique_element_index = unique_map.get(inverse_indices[index], None)

            # Found unique element again, means there's repetition in the solution history
            if unique_element_index is not None:
                return dict(
                    lifespan=index + 1,
                    repetition=index - unique_element_index,
                    start=unique_element_index,
                    end=index
                )
            # New unique value found, add it to the unique map
            else:
                unique_map[inverse_indices[index]] = index

        # No repetition found at all, whole history is unique
        return dict(
            lifespan=solution_history_length,
            repetition=None,
            start=None,
            end=None
        )


if __name__ == '__main__':
    # Disable the logs seagull produces and ignore seagull runtime statistics errors
    logger.disable('seagull')
    filterwarnings('ignore')
    GeneticMethuselahFinder.run_genetic_algorithm(
        Settings.MAX_GENERATIONS,
        Settings.MAX_FITNESS,
        Settings.DIFF_SOLUTION_THRESHOLD
    )
else:
    # Disable the logs seagull produces and ignore seagull runtime statistics errors
    logger.disable('seagull')
    filterwarnings('ignore')
