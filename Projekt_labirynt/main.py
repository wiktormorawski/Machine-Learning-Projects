import random
from deap import base, creator, tools, algorithms

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

items = [(1,1,1,0,1,1,1,0,1,1), (0,0,1,1,1,0,1,0,0,1), (1,1,1,0,1,0,1,1,1,1), (1,0,1,0,0,1,1,1,0,1,1),
             (1,1,0,0,1,1,1,0,1,1), ("lampka nocna", 70, 6), ("srebne sztućce", 100, 1), ("porcelana", 250, 3),
             ("figura z brązu", 300, 10), ("skórzana torebka", 280, 3), ("odkurzacz", 300, 15)
             ]

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=11)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)



def evalOneMax(individual):
    sum_of_weights = 0
    index = 0
    value = 0
    for i in individual:
        if i == 1:
            sum_of_weights += items[index][2]
            value += items[index][1]
            index += 1
        else:
            index += 1
    print(value)
    if sum_of_weights <= 25:
        return value,
    else:
        return 0,

def print_items_chosen(hof):
    index = 0
    items_chosen = []
    for i in hof:
        if i == 1:
            items_chosen.append(items[index][0] + " -> " + str(items[index][1]))
            index += 1
        else:
            index += 1
    return tuple(items_chosen)

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)




def main():
    import numpy

    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.05, ngen=20, stats=stats, halloffame=hof,
                                       verbose=True)

    return pop, logbook, hof


if __name__ == "__main__":
    pop, log, hof = main()
    '''print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))'''
    print(print_items_chosen(hof[0]))
    import matplotlib.pyplot as plt

    gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
    plt.plot(gen, avg, label="average")
    plt.plot(gen, min_, label="minimum")
    plt.plot(gen, max_, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    plt.show()