from experiments.utils import run_experiment
from experiments.baseline import configuration

# --------------------------------------------------
# Activation Function Experiment
# --------------------------------------------------
activations = [
    'sigmoid',
    'relu',
    'tanh',
    'elu'
]

# --------------------------------------------------
# Run Experiments
# --------------------------------------------------
if __name__ == '__main__':
    for activation in activations:
        experiment_config = configuration.copy()

        experiment_config['activation'] = activation

        run_experiment(
            name = f'activation_{activation}',
            configuration = experiment_config
        )
        