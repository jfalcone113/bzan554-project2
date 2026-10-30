from experiments.utils import run_experiment
from experiments.baseline import configuration

# --------------------------------------------------
# Hidden Layers Experiment
# --------------------------------------------------
architectures = {
    '1_layer': [21],
    '2_layers': [21, 12],
    '3_layers': [21, 12, 7],
    '4_layers': [21, 12, 7, 4],
    '5_layers': [21, 12, 7, 4, 2]
}

# --------------------------------------------------
# Run Experiments
# --------------------------------------------------
if __name__ == '__main__':
    for name, hidden_layers in architectures.items():
        experiment_config = configuration.copy()

        experiment_config['hidden_layers'] = hidden_layers

        run_experiment(
            name = f'hidden_layers_{name}',
            configuration = experiment_config
        )