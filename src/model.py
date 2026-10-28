import tensorflow as tf


def build_model(
    input_dim,
    num_skus,
    embedding_dim = 8,
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

    sku_input = tf.keras.Input(
        shape = (1,),
        dtype = 'int32',
        name = 'sku_input'
    )

    features_input = tf.keras.Input(
        shape = (input_dim,),
        name = 'features_input'
    )

    sku_vector = tf.keras.layers.Embedding(
        input_dim = num_skus + 1,
        output_dim = embedding_dim,
        name = 'sku_embedding'
    )(sku_input)

    sku_vector = tf.keras.layers.Flatten()(sku_vector)

    # Unseen SKUs use a zero vector instead of an untrained embedding
    sku_vector = tf.keras.layers.Lambda(
        lambda values: values[0] * tf.cast(
            tf.not_equal(values[1],0),
            values[0].dtype
        ),
        output_shape = (embedding_dim,),
        name = 'mask_unseen_skus'
    )([sku_vector, sku_input])

    x = tf.keras.layers.Concatenate()([
        features_input,
        sku_vector
    ])

    for neurons in hidden_layers:
        x = tf.keras.layers.Dense(
            neurons,
            activation = activation
        )(x)

    output = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.Model(
        inputs = {
            'sku_input': sku_input,
            'features_input': features_input
        },
        outputs = output
    )

    # Optimizer selection
    optimizers = {
        'sgd': tf.keras.optimizers.SGD,
        'adam': tf.keras.optimizers.Adam,
        'rmsprop': tf.keras.optimizers.RMSprop,
        'adagrad': tf.keras.optimizers.Adagrad
    }

    model.compile(
        optimizer = optimizers[optimizer](
            learning_rate = learning_rate
        ),
        loss = 'mse',
        metrics = ['mae']
    )

    return model