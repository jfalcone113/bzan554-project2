import statistics
from numpy import select
import tensorflow as tf

def build_model(
    input_dim,
    hidden_layers = None,
    activation = 'sigmoid',
    optimizer = 'sgd',
    learning_rate = 0.01
):

    # Default architecture
    if hidden_layers is None:
        hidden_layers = [
            max(1, int(input_dim * 0.6)),
            max(1, int(input_dim * 0.6**2)),
            max(1, int(input_dim * 0.6**3))
        ]
    
    # Initialize neural network
    model = tf.keras.Sequential()

    # Input
    model.add(
        tf.keras.layers.Input(shape = (input_dim,))
    )

    # Hidden Layers
    for neurons in hidden_layers:
        model.add(
            tf.keras.layers.Dense(
                units = neurons,
                activation = activation
            )
        )
    
    # Output Layer
    model.add(
        tf.keras.layers.Dense(
            units = 1,
            activation = 'linear'
        )
    )

    # Optimizer selection
    optimizers = {
        'sgd': tf.keras.optimizers.SGD,
        'adam': tf.keras.optimizers.Adam,
        'rmsprop': tf.keras.optimizers.RMSprop,
        'adagrad': tf.keras.optimizers.Adagrad
    }

    selected_optimizer = optimizers[optimizer](
        learning_rate = learning_rate
    )

    # Compile
    model.compile(
        optimizer = selected_optimizer,
        loss = 'mse',
        metrics = ['mae']
    )

    return model

if __name__ == '__main__':
    
    from data import calculate_statistics
    
    statistics = calculate_statistics()
    
    input_dim = len(statistics['categories']) + 3
    
    model = build_model(input_dim)

    model.summary()
