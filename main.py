from niapy.algorithms.basic import GeneticAlgorithm
from niapy.task import Task
from sklearn.datasets import load_iris
from MyProblem import Clustering

for i in range(10):
    task = Task(problem=Clustering(dimension=10), max_evals=100) # 10 dimenzija x
    algorithm = GeneticAlgorithm(population_size=40, crossover_probability=0.9, mutation_probability=0.2)

    best_solution = algorithm.run(task)

    print("Best solution:", best_solution)

