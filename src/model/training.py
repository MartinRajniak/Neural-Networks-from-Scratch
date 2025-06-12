import numpy as np

import copy


def train_model(
    # Data
    inputs_train,
    inputs_test,
    targets_train,
    targets_test,
    # Hyperparams / Configuration / Compile
    epochs,
    batch_size,
    learning_rate,
    # Model
    inputs,
    target,
    loss,
    # Early Stopping
    early_stopping_patience,
    early_stopping_delta,
    # Other
    log=True,
):

    data_size_train = len(inputs_train)
    data_size_test = len(inputs_test)

    min_loss_test = np.inf
    best_loss = None
    best_epoch = 0
    # First epoch, we don't have test loss yet, so we substitute it
    average_loss_test = np.inf
    history = {"train_loss": [], "test_loss": [average_loss_test]}
    indices = np.arange(data_size_train)
    for epoch in range(epochs):
        # Training
        np.random.shuffle(indices)
        losses_train = []
        for batch_index in range(data_size_train // batch_size):
            losses = []
            for sample_index in range(batch_size):
                # Use shuffled index
                index = indices[batch_index * batch_size + sample_index]

                for input_index, input in enumerate(inputs):
                    input.data = inputs_train[index][input_index]
                target.data = targets_train[index]

                loss_value = loss.forward()
                losses.append(loss_value)

            loss_value_train = sum(losses) / len(losses)
            losses_train.append(loss_value_train)

            loss.data = loss_value_train
            loss.backward(learning_rate)

        # Train history for learning curve
        average_loss_train = sum(losses_train) / len(losses_train)
        history["train_loss"].append(average_loss_train)

        # To compare train and test loss on same weights - we need average loss test from last epoch (before backprop)
        if log:
            print(
                f"Epoch {epoch}: loss={average_loss_train}, test_loss={average_loss_test}"
            )

        # Validation
        losses_test = []
        for sample_index in range(data_size_test):
            for input_index, input in enumerate(inputs):
                input.data = inputs_test[sample_index][input_index]
            target.data = targets_test[sample_index]

            loss_value_test = loss.forward()
            losses_test.append(loss_value_test)

        # Test history for learning curve
        average_loss_test = sum(losses_test) / len(losses_test)
        history["test_loss"].append(average_loss_test)

        # Model Checkpoint
        if average_loss_test < min_loss_test:
            min_loss_test = average_loss_test
            best_loss = copy.deepcopy(loss)
            best_epoch = epoch

        # Early Stopping
        train_losses = history["test_loss"]
        if (len(train_losses) >= early_stopping_patience + 1) and (
            (
                min(history["test_loss"][:-early_stopping_patience])
                - min(history["test_loss"][-early_stopping_patience:])
            )
            < early_stopping_delta
        ):
            break

    # Use best model weights
    loss.update_weights(best_loss)

    if log:
        print(f"Best epoch: {best_epoch}")
        print(f"Best test loss: {min_loss_test}")

    return history
