def best_params():
    best = {
        'cb': {
            'lr': 0.07,
            'balance_classes': True,
            'init_prompt': 'entailment contradiction neutral'
        }

        # 'copa': {
        #     'lr': 0.003,
        #     'balance_classes': True,
        #     'init_prompt': ''
        # }

        # 'rte': {
        #     'lr': 0.001,
        #     'balance_classes': True,
        #     'init_prompt': 'entailment contradiction'
        # }

        # 'rte': {
        #     'lr': 0.001,
        #     'balance_classes': True,
        #     'init_prompt': ''
        # }

        # 'boolq': {
        #     'lr': 0.003,
        #     'balance_classes': True,
        #     'init_prompt': ''
        # }
    }

    return best