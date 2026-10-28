from experiments.utils import run_experiment

# --------------------------------------------------
# Baseline Configuration
# --------------------------------------------------

configuration = {
    'epochs': 20,
    'batch_size': 1024,
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'activation': 'sigmoid',
    'random_state': 42,
    'hidden_layers': None,
    'embedding_dim': 8
}

# --------------------------------------------------
# Baseline Execute
# --------------------------------------------------
if __name__ == '__main__':
    run_experiment(
        name = 'baseline',
        configuration = configuration
    )

##### Model Configuration #####
# Hyperparameter Value
# Epochs: 20
# Batch size: 1,024
# Optimizer: Adam
# Learning rate: 0.001
# Activation: Sigmoid
# Hidden layers: 3
# Hidden neurons: [21, 12, 7]
# SKU embedding dimension: 8
# Random seed: 42
# Trainable parameters: 533,559

##### Performance #####
# Metric Training Validation
# Observations: 350,294 75,053
# MSE: 1,190.24 1,612.48
# MAE: 18.81 21.40
# R-squared: 0.5337 0.3852
# Best epoch: 19

