import numpy as np
from data import calculate_statistics, data_generator, numerical_features
from model import build_model

# --------------------------------------------------
# Configuration
# --------------------------------------------------
epochs = 20
batch_size = 1024
optimizer = 'adam'
learning_rate = 0.001
activation = 'sigmoid'
random_state = 42

# --------------------------------------------------
# Training
# --------------------------------------------------
def train_model():
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
    # Incremental Training
    # --------------------------------------------------
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
            for start in range(0, len(X), batch_size):
                end = start + batch_size
                X_batch = X[start:end]
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

        # --------------------------------------------------
        # Store Results
        # --------------------------------------------------

        history['loss'].append(epoch_train_loss)
        history['mae'].append(epoch_train_mae)
        history['loss'].append(epoch_val_loss)
        history['mae'].append(epoch_val_mae)

        print(
            f'Loss: {epoch_train_loss:.6f} |'
            f'MAE: {epoch_train_mae:.6f} |'
            f'Val Loss: {epoch_val_loss:.6f} |'
            f'Val MAE: {epoch_val_mae:.6f} |'
        )
    
    return model, history, statistics

# --------------------------------------------------
# Execute Training
# --------------------------------------------------
if __name__ == '__main__':
    model, history, statistics = train_model()
