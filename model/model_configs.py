neumf_config = {
    # 'num_users': None,
    # 'num_items': None,
    'latent_dim_mf': 64,
    'latent_dim_mlp': 64,
    # 'num_negative': 4,
    'layers': [128, 64, 32, 16],  # layers[0] is the concat of latent user vector & latent item vector
}

mf_config = {
    'latent_dim_mf':64,
}

LightGCN_config = {
    'in_size': 32,
    'layer_size': [64, 64, 64],
    'dropout': [0.1, 0.1, 0.1],
}
