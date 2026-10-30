from experiments.utils import run_experiment
from experiments.baseline import configuration

# --------------------------------------------------
# Hidden Neurons Experiment
# --------------------------------------------------

architectures = {
    '8_neurons': [8, 8, 8],
    '16_neurons': [16, 16, 16],
    '32_neurons': [32, 32, 32],
    '64_neurons': [64, 64, 64],
    '128_neurons': [128, 128, 128]
    }

# --------------------------------------------------
# Run Experiment
# --------------------------------------------------
if __name__ == '__main__':
    for name, hidden_layers in architectures.items():
        experiment_config = configuration.copy()

        experiment_config['hidden_layers'] = hidden_layers

        run_experiment(
            name = f'hidden_neurons_{name}',
            configuration = experiment_config
        )