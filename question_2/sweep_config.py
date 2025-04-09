sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'filters': {'values': [16, 32, 64]},
        'dropout': {'values': [0.2, 0.3]},
        'activation': {'values': ['relu', 'tanh']},
        'use_batchnorm': {'values': [True, False]},
        'padding': {'values': ['same', 'valid']},
        'epochs': {'value': 5},
        'lr': {'values': [0.001, 0.0005]}
    }
}
