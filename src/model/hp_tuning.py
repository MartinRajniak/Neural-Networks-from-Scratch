from itertools import product

import numpy as np

def start_hp_search(hp_grid, train_and_evaluate):
    iteration = 0
    best_loss = np.inf
    best_hp_config = None

    for hp_values in product(*hp_grid.values()):
        hp_config = {key: value for key, value in zip(hp_grid.keys(), hp_values)}

        iteration += 1
        print(f"Search iteration: {iteration}")
        print(f"Testing config: {hp_config}")

        average_loss = train_and_evaluate(hp_config)

        print(f"Loss: {average_loss}")

        if (average_loss < best_loss):
            best_loss = average_loss
            best_hp_config = hp_config
    
    return best_hp_config