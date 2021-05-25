import math
import random
from deap import base, creator, tools, algorithms
import numpy
import time

def genetic_first_Fitness(inp, popul, gener, elitism):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    labirynth = numpy.array([(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                             (0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0),
                             (0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0),
                             (0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0),
                             (0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0),
                             (0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0),
                             (0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0),
                             (0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0),
                             (0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0),
                             (0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0),
                             (0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 10, 0),
                             (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                             ])
    # labirynt 10 na 10
    # 1 - lewo   2 - prawo   3 - góra   4 - dół
    # Randomowe liczby 1 2 3 4
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.choice, [1, 2, 3, 4])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=inp)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def fitness_first(individual):
        def calc_meta_dist(position):
            return abs(10 - position[0]) + abs(10 - position[1])

        def check_if_on_bounds(position):
            if labirynth[position[1]][position[0]] == 1:
                return False
            else:
                return True

        current_possition = [1, 1]
        steps = 0
        for choice in individual:
            steps += 1
            if choice == 1:
                current_possition[0] -= 1
            if choice == 2:
                current_possition[0] += 1
            if choice == 3:
                current_possition[1] -= 1
            if choice == 4:
                current_possition[1] += 1
            if current_possition[0] < 1 or current_possition[1] < 1:
                return (18 - calc_meta_dist(current_possition)) / 18
            if check_if_on_bounds(current_possition):
                if (18 - calc_meta_dist(current_possition)) / 18 == 1:
                    return (18 - calc_meta_dist(current_possition)) / 18 + 1 / steps
                return (18 - calc_meta_dist(current_possition)) / 18
            if current_possition == [10, 10]:
                return (1 + 20 / steps)
        return (((18 - calc_meta_dist(current_possition)) / 18))

    def evalOneMax(individual):
        return fitness_first(individual),

    def print_steps_chosen(hof):
        lista = []
        for i in hof:
            if i == 1:
                lista.append("lewo")
            if i == 2:
                lista.append("prawo")
            if i == 3:
                lista.append("góra")
            if i == 4:
                lista.append("dół")
        return lista

    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def main():
        pop = toolbox.population(n=popul)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
        if elitism:
            # mu to ile ma przechodzic do nastepnej generacji, a lambda to ile populacji wytwarzam z mu !
            pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=gener, stats=stats,lambda_=popul,mu=int(popul/10), halloffame=hof,
                                           verbose=True)

        else:
            pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=gener, stats=stats, halloffame=hof,
                                           verbose=True)

        return pop, logbook, hof

    pop, log, hof = main()
    print(print_steps_chosen(hof[0]))



'''-----------------------------------------------------------------------------------------------'''


def genetic_second_Fitness(inp, popul, gener, elitism):
    inputsize = inp

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    labirynth = numpy.array([(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                             (0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0),
                             (0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0),
                             (0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0),
                             (0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0),
                             (0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0),
                             (0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0),
                             (0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0),
                             (0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0),
                             (0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0),
                             (0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 10, 0),
                             (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                             ])
    # labirynt 10 na 10
    # 1 - lewo   2 - prawo   3 - góra   4 - dół
    # Randomowe liczby 1 2 3 4
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.choice, [1, 2, 3, 4])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=inputsize)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def fitness_second(individual):
        def calc_meta_dist(x, y):
            return math.sqrt((10 - x) ** 2 + (10 - y) ** 2)

        steps = 0
        i = 0
        x = 1
        y = 1
        while i < inputsize:
            steps += 1
            element = individual[i]
            if element == 1:
                x -= 1
            if element == 2:
                x += 1
            if element == 3:
                y -= 1
            if element == 4:
                y += 1
            if y == 0 or y == 11:
                return 0
            if x == 0 or x == 11:
                return 0
            if labirynth[y][x] == 0:
                return (18 - calc_meta_dist(x, y)) / 18
            if x == 10 and y == 10:
                return 1
            i += 1
        return (18 - calc_meta_dist(x, y)) / 18

    def evalOneMax(individual):
        return fitness_second(individual),

    def print_steps_chosen(hof):
        lista = []
        for i in hof:
            if i == 1:
                lista.append("lewo")
            if i == 2:
                lista.append("prawo")
            if i == 3:
                lista.append("góra")
            if i == 4:
                lista.append("dół")
        return lista

    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def main():
        pop = toolbox.population(n=popul)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
        if elitism:
            # mu to ile ma przechodzic do nastepnej generacji, a lambda to ile populacji wytwarzam z mu !
            pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=gener, stats=stats,
                                                     lambda_=popul, mu=int(popul / 10), halloffame=hof,
                                                     verbose=True)

        else:
            pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=gener, stats=stats,
                                               halloffame=hof,
                                               verbose=True)
        return pop, logbook, hof

    pop, log, hof = main()
    print(print_steps_chosen(hof[0]))
    # gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
    # plt.plot(gen, avg, label="average")
    # plt.plot(gen, min_, label="minimum")
    # plt.plot(gen, max_, label="maximum")
    # plt.xlabel("Generation")
    # plt.ylabel("Fitness")
    # plt.legend(loc="lower right")
    # plt.show()


'''------------------------------------------------------------------------------------'''

# Żródło to : https://www.geeksforgeeks.org/building-an-undirected-graph-and-finding-shortest-path-using-dictionaries-in-python/
# lista = ([(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#           (0, 'A ', 'B ', 'C ', 0, 'D ', 'E ', 'F ', 0, 'G ', 'H ', 0),
#           (0, 0, 0, 'I ', 'J ', 'K ', 0, 'L ', 0, 0, 'M ', 0),
#           (0, 'N ', 'P ', 'R ', 0, 'S ', 0, 'T', 'U', 'W', 'Z', 0),
#           (0, 'A1', 0, 'B1', 0, 0, 'C1', 'D1', 0, 'E1', 'F1', 0),
#           (0, 'G1', 'H1', 0, 0, 'I1', 'J1', 'K1', 0, 'L1', 'M1', 0),
#           (0, 'N1', 'P1', 'R1', 'S1', 'T1', 0, 'U1', 'W1', 'Z1', 0, 0),
#           (0, 'A2', 0, 'B2', 'C2', 0, 0, 'D2', 0, 'E2', 'F2', 0),
#           (0, 'G2', 0, 0, 0, 'H2', 'I2', 'J2', 0, 0, 'K2', 0),
#           (0, 'L2', 0, 'M2', 0, 0, 'N2', 0, 'P2', 0, 'R2', 0),
#           (0, 'S2', 0, 'T2', 'U2', 'W2', 'Z2', 'A3', 'B3', 'C3', 'META', 0),
#           (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
#           ])
# lista = (['0 ', '0 ', '0 ', '0 ', '0 ', '0 ', '0 ', '0 ', '0 ', '0 ', '0 ', '0 ']
# ['0 ', '1 ', '2 ', '3 ', '0 ', '4 ', '5 ', '6 ', '0 ', '7 ', '8 ', '0 ']
# ['0 ', '0 ', '0 ', '9 ', '10 ', '11 ', '0 ', '12 ', '0 ', '0 ', '13 ', '0 ']
# ['0 ', '14 ', '15 ', '16 ', '0 ', '17 ', '0 ', '18 ', '19 ', '20 ', '21 ', '0 ']
# ['0 ', '22', '0 ', '23', '0 ', '0 ', '24', '25', '0 ', '26', '27', '0 ']
# ['0 ', '28', '29', '0 ', '0 ', '', '30', '31', '0 ', '32', '33', '0 ']
# ['0 ', '34', '35', '36', '37', '38', '0 ', '39', '40', '41', '0 ', '0 ']
# ['0 ', '42', '0 ', '43', '44', '0 ', '0 ', '45', '0 ', '46', '47', '0 ']
# ['0 ', '48', '0 ', '0 ', '0 ', '49', '50', '51', '0 ', '0 ', '52', '0 ']
# ['0 ', '53', '0 ', '54', '0 ', '0 ', '55', '0 ', '56', '0 ', '57', '0 ']
# ['0 ', '58', '0 ', '59', '60', '61', '62', '63', '64', '65', '66', '0 ']
# ['0 ', '0 ', '0 ', '0 ', '0 ', '0 ', '0 ', '0 ', '0 ', '0 ', '0 ', '0 '])
graph = {'1': ['2'],
         '2': ['3'],
         '3': ['9'],
         '4': ['11', '5'],
         '5': ['6', '4'],
         '6': ['5', '12'],
         '7': ['8'],
         '8': ['13', '7'],
         '9': ['3', '10', '16'],
         '10': ['11', '9'],
         '11': ['17', '10', '4'],
         '12': ['6', '18'],
         '13': ['21', '8'],
         '14': ['22', '15'],
         '15': ['16', '14'],
         '16': ['15', '23', '9'],
         '17': ['11'],
         '18': ['12', '25', '19'],
         '19': ['18', '20'],
         '20': ['21', '19'],
         '21': ['13', '20', '26'],
         '22': ['14', '27'],
         '23': ['16'],
         '24': ['25', '30'],
         '25': ['18', '24', '31'],
         '26': ['21', '33'],
         '27': ['22', '34', '28'],
         '28': ['35', '27'],
         '29': ['38', '30'],
         '30': ['24', '31', '29'],
         '31': ['30', '25', '39'],
         '32': ['41', '33'],
         '33': ['26', '32'],
         '34': ['27', '35', '42'],
         '35': ['28', '36', '34'],
         '36': ['35', '43', '37'],
         '37': ['36', '44', '38'],
         '38': ['29', '37'],
         '39': ['31', '40', '45'],
         '40': ['41', '39'],
         '41': ['40', '32', '46'],
         '42': ['34', '48'],
         '43': ['36', '44'],
         '44': ['37', '43'],
         '45': ['39', '51'],
         '46': ['41', '47'],
         '47': ['46', '52'],
         '48': ['42', '53'],
         '49': ['50'],
         '50': ['55', '51', '49'],
         '51': ['45', '50'],
         '52': ['47', '57'],
         '53': ['48', '58'],
         '54': ['59'],
         '55': ['50', '62'],
         '56': ['64'],
         '57': ['52', '66'],
         '58': ['53'],
         '59': ['54', '60'],
         '60': ['59', '61'],
         '61': ['60', '62'],
         '62': ['55', '61', '63'],
         '63': ['62', '64'],
         '64': ['56', '63', '65'],
         '65': ['64', '66'],
         '66': ['57', '65'],
         }


# ll = []
# for i in lista:
#     new_list = []
#     for l in i:
#         if type(l) == str:
#             if len(l) > 1:
#                 new_list.append(l)
#             else:
#                 new_list.append(l + ' ')
#         if len(str(l)) == 1:
#             new_list.append(str(l) + ' ')
#
#     ll.append(new_list)
# for i in ll:
#     print(i)


def BFS_SP(graph, start, goal):
    explored = []
    queue = [[start]]

    if start == goal:
        print("Same Node")
        return

    # Loop to traverse the graph
    # with the help of the queue
    while queue:
        path = queue.pop(0)
        node = path[-1]

        # Codition to check if the
        # current node is not visited
        if node not in explored:
            neighbours = graph[node]

            # Loop to iterate over the
            # neighbours of the node
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)

                # Condition to check if the
                # neighbour node is the goal
                if neighbour == goal:
                    print("Shortest path = ", *new_path)
                    return new_path
            explored.append(node)

    # Condition when the nodes
    # are not connected
    print("So sorry, but a connecting path doesn't exist :(")
    return


def return_time_of_bfs():
    start = time.time()
    BFS_SP(graph, '1', '66')
    return time.time() - start
# Żródło WYKRESY https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

def return_time_with_1_fitness(inp, pop, gen, eli):
    start = time.time()
    genetic_first_Fitness(inp, pop, gen, eli)
    return time.time() - start
def return_time_with_2_fitness(inp, pop, gen, eli):
    start = time.time()
    genetic_second_Fitness(inp, pop, gen, eli)
    return time.time() - start

genetic_first_Fitness_20_500_200_false_time = return_time_with_1_fitness(20, 500, 200, False)
genetic_first_Fitness_20_500_200_true_time = return_time_with_1_fitness(20, 500, 200, True)
genetic_first_Fitness_40_2000_1000_false_time = return_time_with_1_fitness(40, 2000, 1000, False)
genetic_first_Fitness_40_2000_1000_true_time = return_time_with_1_fitness(40, 2000, 1000, True)

bfs_time = return_time_of_bfs()

genetic_second_Fitness_20_500_200_false_time = return_time_with_2_fitness(20, 500, 200, False)
genetic_second_Fitness_20_500_200_true_time = return_time_with_2_fitness(20, 500, 200, True)
genetic_second_Fitness_40_2000_1000_false_time = return_time_with_2_fitness(40, 2000, 1000, False)
genetic_second_Fitness_40_2000_1000_true_time = return_time_with_2_fitness(40, 2000, 1000, True)

import matplotlib.pyplot as plt
import numpy as np


labels = ['G1(20)', 'G1(40)', 'G2(20)', 'G2(40)', 'BFS']
men_means = [genetic_first_Fitness_20_500_200_false_time, genetic_first_Fitness_40_2000_1000_false_time, genetic_second_Fitness_20_500_200_false_time, genetic_second_Fitness_40_2000_1000_false_time, bfs_time]
women_means = [genetic_first_Fitness_20_500_200_true_time, genetic_first_Fitness_40_2000_1000_true_time, genetic_second_Fitness_20_500_200_true_time, genetic_second_Fitness_40_2000_1000_true_time, bfs_time]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Z Elityzmem')
rects2 = ax.bar(x + width/2, women_means, width, label='Bez Elityzmu')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Czas działania')
ax.set_title('Zestawienie Algorytmow genetycznych i BFS')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()
