from src.model.variable import Variable

import numpy as np

import functools

from typing import List

def neuron(inputs: List, use_activation = False, n_outputs = 1):
    bias = Variable(label="bias", data=0.01, trainable=True)

    # Xavier/Glorot initialization (simplified since output is 1)
    scale = np.sqrt(6.0 / len(inputs) + n_outputs)
    weighted_inputs = []
    for index, input in enumerate(inputs):
        weight = Variable(label=f"weight{index}", data=np.random.uniform(-scale, scale), trainable=True)
        weighted_input = input * weight
        weighted_input.label = f"weighted_input{index}"
        weighted_inputs.append(weighted_input)

    weighted_input_sum = functools.reduce(lambda x, y: x + y, weighted_inputs)
    weighted_input_sum.label = "weighted_input_sum"

    biased_sum = weighted_input_sum + bias
    biased_sum.label = "sum"

    if (use_activation):
        output = biased_sum.tanh()
    else:
        output = biased_sum
    output.label = "output"

    return output