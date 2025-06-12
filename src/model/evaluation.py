import numpy as np

def evaluate_model(inputs_test, targets_test, inputs, target, loss):
    losses = []
    for sample_index in range(len(inputs_test)):
        for input_index, input in enumerate(inputs):
            input.data = inputs_test[sample_index][input_index]
        target.data = targets_test[sample_index]
        loss_value = loss.forward()
        losses.append(loss_value)

    return np.mean(losses)