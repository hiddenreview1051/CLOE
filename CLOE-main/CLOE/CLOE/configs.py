model_config_CLOE = {
    'epochs-pre': 10,
    'epochs-joint': 150,
    'learning-rate': 1e-4,
    'data-dir': 'CLOE/datasets/',
    'dim': [500, 500, 2000],
    'num-workers': 0,
    'lambda-CLOE': 1,
    'patience': 20,
    'nb-class': 8,
    'type-conc': 'mean',
    'umap': False,
}