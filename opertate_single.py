from main import optimize
import wandb
from datetime import datetime
import random

use_wandb = True
project = 'test_' + datetime.now().strftime("%m%d_%H%M")
fwd_model = 'LSTM'
fwd_path = "./fwdnet/net/241112/LSTM_Txy_lr_0.001894291444441104_bsize_500_init_X_N_fc_feature_layer_100_300_500_500_500_hidden_size_500_nlayer_3/MSE=0.000735.pt"
data_dir = './fwdnet/gen_20241111/preprocessed/'
nk_dir = './fwdnet/nk_data/241024/interpolated-npy/'

args = {
    'init': 'random',
    'sweep': False,
    
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
    'ct_weight': [0, 1e-5],
    # 'ct_weight': [5e3, 1e-5],
    'norm_weight': 1e2,
    'ssim_weight': 1e2,
    # 'eff_weight': 0.05,
    'eff_weight': 0,
    'edge_weight': 1e3,

    # Loop arguments
    'steps': 500,
    'step_term': 1,

    # Metasurface arguments
    'num_meta': 400,
    'wl': [450.0e-9, 532.0e-9, 635.0e-9],
    
    # Optimization arguments
    'h_lr': 0.001830563486981576,
    'lw_lr': 0.008709137504741126,
    'mat_lr': 0.0018708121087584791,
    'numP_lr': 0.00301652174345815,
    'p_lr': 0.00262933151723463,
    'rot_lr': 0.009752104156246217,
    # 'mat_lr': 3e-3,
    # 'h_lr': 3e-2,
    # 'p_lr': 1e-2,
    # 'lw_lr': 3e-2,
    # 'numP_lr': 3e-2,
    # 'rot_lr': 3e-2,

    # Etc
    'seed': random.randint(0, 1000),
    'ar_lim': 20.0,
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

if use_wandb:
    wandb_args = {}
    args = dict(args, **wandb_args)
    wandb.init(entity='linkle115',
               project=project,
               config=args
               )
    
optimize(args)