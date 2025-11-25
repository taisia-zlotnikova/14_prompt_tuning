# hyperparameters.py

def get_hyperparameter_grid(task_name):
    """
    ПОЛНЫЙ перебор всех комбинаций гиперпараметров
    """
    
    # Все значения learning rate
    base_lrs = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
    
    # Обе опции балансировки
    balance_options = [True, False]
    
    # Все промпты для каждой задачи
    task_specific_prompts = {
        'cb': [
            '',
            'Classify the relationship:',
            'entailment, contradiction, or neutral?',
        ],
        'boolq': [
            '',
            'Based on the text, yes or no?', 
            'Question answering:',
            'yes no:'
        ],
        'copa': [
            '',
            'Choose the correct cause/effect:',
            'first, second',
        ],
        'rte': [
            '',
            'Does the premise support the hypothesis?',
            'entailment or not:',
        ],
        'wic': [
            '',
            'Same meaning in both contexts?',
            'same or different',
        ]
    }
    
    # ПОЛНЫЙ ПЕРЕБОР ВСЕХ КОМБИНАЦИЙ
    grid = []
    
    for lr in base_lrs:
        for balance in balance_options:
            for prompt in task_specific_prompts.get(task_name, ['']):
                grid.append({
                    'lr': lr,
                    'balance_classes': balance,
                    'init_prompt': prompt
                })
    
    total_combinations = len(grid)
    print(f"🔧 Generated FULL grid: {total_combinations} configurations for {task_name}")
    print(f"   (LRs: {len(base_lrs)}, Balance: {len(balance_options)}, Prompts: {len(task_specific_prompts.get(task_name, ['']))})")
    
    return grid

def get_quick_grid(task_name):
    """
    Быстрая сетка для отладки - минимальный набор
    """
    return [
        {'lr': 0.001, 'balance_classes': True, 'init_prompt': ''},
        {'lr': 0.01, 'balance_classes': False, 'init_prompt': 'Classify:'},
    ]

# Предустановленные конфигурации для быстрого старта
PRESET_CONFIGS = {
    'cb': [
        {'lr': 0.003, 'balance_classes': True, 'init_prompt': 'Classify the relationship:'},
        {'lr': 0.01, 'balance_classes': False, 'init_prompt': 'Textual entailment:'},
    ],
    'boolq': [
        {'lr': 0.001, 'balance_classes': True, 'init_prompt': 'Answer the question:'},
        {'lr': 0.003, 'balance_classes': False, 'init_prompt': 'Based on the text:'},
    ],
    'copa': [
        {'lr': 0.01, 'balance_classes': True, 'init_prompt': 'Commonsense reasoning:'},
        {'lr': 0.03, 'balance_classes': False, 'init_prompt': 'Choose the correct option:'},
    ],
    'rte': [
        {'lr': 0.003, 'balance_classes': True, 'init_prompt': 'Inference classification:'},
        {'lr': 0.01, 'balance_classes': False, 'init_prompt': 'True or false?'},
    ],
    'wic': [
        {'lr': 0.001, 'balance_classes': True, 'init_prompt': 'Word sense disambiguation:'},
        {'lr': 0.003, 'balance_classes': False, 'init_prompt': 'Same meaning?'},
    ]
}