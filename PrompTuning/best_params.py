def best_params():
    best = {
        'cb': {
            'lr': 0.07,
            'balance_classes': True,
            'init_prompt': 'Classify the relationship:'
        },

        'copa': {
            'lr': 0.003,
            'balance_classes': True,
            'init_prompt': 'Choose the correct cause/effect:'
        },

        'rte': {
            'lr': 0.001,
            'balance_classes': True,
            'init_prompt': 'Does the premise support the hypothesis?'
        }
    }

    return best