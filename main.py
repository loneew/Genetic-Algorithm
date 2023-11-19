import numpy as np
import pygad
import time
from matplotlib import pyplot as plt

sudoku_size = 9
number_of_squares_in_a_line = int(np.sqrt(sudoku_size))
sudoku = np.zeros((sudoku_size, sudoku_size))
error = []  # змінна для збереження помилок


def get_score(array):  # кількість унікальних значень та призначення балів за кожен унікальний елемент у рядку, колонці чи квадраті.
    array_converted = array.tolist()
    score = 0
    array_error = 0

    for element in array_converted:
        count = array_converted.count(element)
        array_error += count
        if count == 1:
            score += 1
    error.append(array_error)
    return score


def fitness_function(ga_instance, solution, solution_index):
    sudoku_chromosome = np.array(solution).reshape((sudoku_size, sudoku_size))  # Розв'язок генетичного алгоритму
    fitness_value = 0

    for row_and_col in range(sudoku_size):  # Підрахунок балів за рядками та колонками
        fitness_value += get_score(sudoku_chromosome[row_and_col])
        fitness_value += get_score(sudoku_chromosome.T[row_and_col])

    matrix_cutted_in_cols = sudoku_chromosome.reshape(  # Розбиття матриці на квадрати
        (number_of_squares_in_a_line, number_of_squares_in_a_line, sudoku_size))

    for cutted_row in matrix_cutted_in_cols:  # Транспонування квадратів
        transposed_squares = np.split(cutted_row.T, number_of_squares_in_a_line)

        for transposed_square in transposed_squares:  # Розбиття на рядки та виклик функції get_score
            square_list = transposed_square.reshape(sudoku_size)
            fitness_value += get_score(square_list)

    return fitness_value


def on_generation(ga_instance):
    if ga_instance.generations_completed % 100 == 0:
        print(f"Generation {ga_instance.generations_completed}, Best Fitness: {ga_instance.best_solution()[1]}")


mutation_types = ["random", "swap", "adaptive"]  # різні рівні мутації
parent_selection_types = ["sss", "tournament"]  # різні стратегії відбору батьків для схрещування
crossover_types = ["two_points", "uniform"]  # різні варіанти схрещування
i = 1
for mutation_type in mutation_types:
    for parent_selection_type in parent_selection_types:
        for crossover_type in crossover_types:
            start_time = time.time()
            if mutation_type == "adaptive":
                mutation_probability = [0.3, 0.1]
            else:
                mutation_probability = 0.2
            ga_params = {
                "num_generations": 400,
                "num_parents_mating": 2,
                "sol_per_pop": 30,
                "num_genes": sudoku_size * sudoku_size,
                "gene_space": np.arange(1, sudoku_size + 1),
                "parent_selection_type": parent_selection_type,
                "keep_parents": 2,
                "crossover_type": crossover_type,
                "crossover_probability": 0.3,
                "mutation_type": mutation_type,
                "mutation_probability": mutation_probability,
                "stop_criteria": ["saturate_360", f"reach_{3 * sudoku_size ** 2}"],
                "gene_type": int,
                "save_solutions": True,
                "save_best_solutions": False,
                "random_seed": 45,
                "random_mutation_min_val": 1,
                "random_mutation_max_val": sudoku_size,
                "parallel_processing": 4
                # ,"on_generation": on_generation,
            }

            ga_instance = pygad.GA(**ga_params, fitness_func=fitness_function)
            ga_instance.run()
            best_solution, best_fitness, best_solution_idx = ga_instance.best_solution()

            end_time = time.time()
            execution_time = end_time - start_time

            print(f"\t{i}) Mutation Type: {mutation_type}\n\tParent Selection Type: {parent_selection_type}\n"
                  f"\tCrossover Type: {crossover_type}")
            print(f"best Fitness Value: {best_fitness}")
            print(f"mean error per cell: {round(np.mean(error) / (sudoku_size ** 2), 2)}")
            print(f"execution time: {round(execution_time, 2)} seconds")
            print("sudoku solution:")
            print(np.array(best_solution).reshape((sudoku_size, sudoku_size)))
            ga_instance.plot_fitness(
                title=f"{i}. Mutation type: {mutation_type};\nParent selection type: {parent_selection_type};\nCrossover type: {crossover_type}")
            ga_instance.plot_new_solution_rate(
                title=f"{i}. Mutation type: {mutation_type};\nParent selection type: {parent_selection_type};\nCrossover type: {crossover_type}",
                plot_type="bar")

            plot_error = np.array(error)[-81:-1]  # останні 80 значень
            x_values = list(range(0, len(plot_error)))  # або x_values = np.arange(0, len(plot_error))
            plt.plot(x_values, plot_error, '-o')
            plt.xlabel('Ітерація')
            plt.ylabel('Значення помилки')
            plt.title(
                f"{i}. Mutation type: {mutation_type};\nParent selection type: {parent_selection_type};\nCrossover type: {crossover_type}")
            plt.show()

            print("\n")
            error.clear()
            i += 1
