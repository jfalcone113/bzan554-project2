import tensorflow as tf

from .data import calculate_statistics, data_generator, numerical_features
from .model import build_model
from .tensorboard_logger import create_writer, log_metrics


# --------------------------------------------------
# Training
# --------------------------------------------------
def train_model(
    epochs = 20,
    batch_size = 1024,
    optimizer = 'adam',
    learning_rate = 0.001,
    activation = 'sigmoid',
    random_state = 42,
    hidden_layers = None,
    embedding_dim = 8,
    run_id = None
):
    tf.keras.utils.set_random_seed(random_state)
    # Calculate training statistics
    print('Calculating training statistics...')
    statistics = calculate_statistics()

    # Determine input dimensions
    input_dim = (
        len(statistics['categories'])
        + len(numerical_features)
    )

    # Initialize model
    model = build_model(
        input_dim = input_dim,
        num_skus = len(statistics['sku_lookup']),
        embedding_dim = embedding_dim,
        hidden_layers = hidden_layers,
        activation = activation,
        optimizer = optimizer,
        learning_rate = learning_rate
    )

    model.summary()

    # Store training history
    history = {
        'loss':[],
        'mae':[],
        'val_loss':[],
        'val_mae':[]
    }
    
    # --------------------------------------------------
    # Experiment Tracking
    # --------------------------------------------------
    if run_id is None:

        architecture = (
            '-'.join(map(str, hidden_layers))
            if hidden_layers is not None
            else 'default'
        )

        run_name = (
            f'{optimizer}_'
            f'{activation}_'
            f'{learning_rate}_'
            f'{architecture}_'
            f'{embedding_dim}'
        )
    
    else:
        run_name = run_id

    # --------------------------------------------------
    # Incremental Training
    # --------------------------------------------------
    writer = create_writer(
        run_name = run_name,
        add_timestamp = run_id is None
        )

    best_val_loss = float('inf')
    best_epoch = 0
    best_weights = None

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        model.reset_metrics()

        for X, y in data_generator(
            'train',
            statistics,
            shuffle = True,
            seed = random_state + epoch
        ):
            # Divide chunk into mini batches
            for start in range(0, len(y), batch_size):
                end = start + batch_size

                X_batch = {
                    name: values[start:end]
                    for name, values in X.items()
                }
                y_batch = y[start:end]

                # Update weights using mini batch
                train_metrics = model.train_on_batch(
                    X_batch,
                    y_batch,
                    return_dict = True
                )

        # Training metrics
        epoch_train_loss = train_metrics['loss']
        epoch_train_mae = train_metrics['mae']

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
        model.reset_metrics()

        for X, y in data_generator(
            'val',
            statistics,
            shuffle = False
        ):
            val_metrics = model.test_on_batch(
                X,
                y,
                return_dict = True
            )
    
        # Validation epochs
        epoch_val_loss = val_metrics['loss']
        epoch_val_mae = val_metrics['mae']

        # Track best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            best_weights = model.get_weights()

        # --------------------------------------------------
        # Store Results
        # --------------------------------------------------

        history['loss'].append(epoch_train_loss)
        history['mae'].append(epoch_train_mae)
        history['val_loss'].append(epoch_val_loss)
        history['val_mae'].append(epoch_val_mae)

        log_metrics(
            writer = writer,
            epoch = epoch + 1,
            train_loss = epoch_train_loss,
            val_loss = epoch_val_loss,
            train_mae = epoch_train_mae,
            val_mae = epoch_val_mae
        )

        print(
            f'Loss: {epoch_train_loss:.6f} |'
            f'MAE: {epoch_train_mae:.6f} |'
            f'Val Loss: {epoch_val_loss:.6f} |'
            f'Val MAE: {epoch_val_mae:.6f} |'
        )
    
    if best_weights is not None:
        model.set_weights(best_weights)

    print(f'\nBest Epoch: {best_epoch}')
    print(f'\nBest Validation Loss: {best_val_loss:.6f}')

    writer.close()
    return model, history, statistics

# --------------------------------------------------
# Execute Training
# --------------------------------------------------
if __name__ == '__main__':
    from .evaluate import evaluate_model
    
    model, history, statistics = train_model(epochs = 20)

    for split in ['train','val']:
        metrics = evaluate_model(model, statistics, split)
        print(f'\n{split.upper()}: {metrics}')
    