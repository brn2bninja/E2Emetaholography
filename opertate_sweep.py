from main import optimize
import wandb
import random

use_wandb = True
project="sweep_lr"
fwd_model = 'LSTM'
fwd_path =  '/home/linkle115/vscode_server/EEasih/fwdnet/net/241112/LSTM_Txy_lr_0.001894291444441104_bsize_500_init_X_N_fc_feature_layer_100_300_500_500_500_hidden_size_500_nlayer_3/MSE=0.000735.pt'
data_dir =  '/home/linkle115/vscode_server/EEasih/fwdnet/gen_20241111/preprocessed/'
nk_dir =    '/home/linkle115/vscode_server/EEasih/fwdnet/nk_data/241024/interpolated-npy/'

args = {
    'init': 'random',
    'sweep': True,
    
    # WandB configurations
    'wandb': use_wandb,
    'project': project,

    # Target image
    'target_img': 'logo2',

    # Foward network arguments
    'fwd_path': fwd_path,
    'fwd_model': fwd_model,
    'data_dir': data_dir,
    'nk_dir': nk_dir,

    # Loss arguments
    'color_weights': [1.0, 1.0, 1.0],
    'bce_weight': 0.0,
    'ct_weight': [0, 0],
    'norm_weight': 100.0,
    'ssim_weight': 0.0,
    'eff_weight': 0,
    'edge_weight': 1e3,
    
    # Loop arguments
    'steps': 200,
    'step_term': 10,

    # Metasurface arguments
    'num_meta': 400,
    'wl': [450.0e-9, 532.0e-9, 635.0e-9],

    # Etc
    'seed': random.randint(0, 1000),
    'ar_lim': 20.0,
    'path_holo': './fwdnet/tri_hologram_450/metaatom_dictionaries/meta_dict_nk_T100P30h=880_p=400_ar-100_num-700.pkl',
    }

if fwd_model == "LSTM":
    fwdnet_file = fwd_path.split('/')[-2]
    init = fwdnet_file.split('init_')[1][:3]
    if fwdnet_file.find('hidden_size') != -1:
        hs = int(fwdnet_file.split('hidden_size_')[1].split('_nlayer')[0])
        channels = [int(i) for i in fwdnet_file.split('fc_feature_layer_')[1].split('_hidden_size')[0].split('_')]
    else:
        hs = int(fwdnet_file.split('hid_size_')[1].split('_nlayer')[0])
        channels = [int(i) for i in fwdnet_file.split('fc_feature_layer_')[1].split('_hid_size')[0].split('_')]
    nl = int(fwdnet_file.split('_nlayer_')[1])
    fc_layers = [{'in_channels': channels[i], 'out_channels': channels[i+1]} for i in range(len(channels)-1)]
    additional_args = {'init': init,
                     'input_size': 7,
                     'hidden_size': hs,
                     'num_layer': nl,
                     'fc_feature_layer': fc_layers}

args['additional_args'] = additional_args

sweep_configuration = {
        "entity": "linkle115",
        "method": "bayes",
        "name": "Optimal_lrs",
        "metric": {"goal": "minimize",
                   "name": "Total_loss"},
        "parameters": {
            "mat_lr":   {"max": 1e-2,   "min": 3e-4,    'distribution': 'log_uniform_values'},
            "h_lr":     {"max": 1e-2,   "min": 3e-4,    'distribution': 'log_uniform_values'},
            "p_lr":     {"max": 1e-2,   "min": 3e-4,    'distribution': 'log_uniform_values'},
            "lw_lr":    {"max": 1e-2,   "min": 3e-4,    'distribution': 'log_uniform_values'},
            "numP_lr":  {"max": 1e-2,   "min": 3e-4,    'distribution': 'log_uniform_values'},
            "rot_lr":   {"max": 1e-2,   "min": 3e-4,    'distribution': 'log_uniform_values'},
        },
        "early_terminate": {
            "type": "hyperband",
            "s": 2,
            "eta": 3,
            "max_iter": 5,    
        },
    }

def sweep_main():
    wandb.init()
    wandb_args = {}
    for key in sweep_configuration['parameters'].keys():
        wandb_args[key] = getattr(wandb.config, key)
    
    args['mat_lr']  = wandb_args["mat_lr"]
    args['h_lr']    = wandb_args["h_lr"]
    args['p_lr']    = wandb_args["p_lr"]
    args['lw_lr']   = wandb_args["lw_lr"]
    args['numP_lr'] = wandb_args["numP_lr"]
    args['rot_lr']  = wandb_args["rot_lr"]
        
    wandb.init(entity='linkle115',
                project=project,
                config=wandb_args
                )
    optimize(args)


sweep_id = wandb.sweep(sweep_configuration, project=project)
wandb.agent(sweep_id, function=sweep_main)
