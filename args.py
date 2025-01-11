# Argument parameters file

import argparse

def parse_args():
    def str2bool(v):
        assert(v == 'True' or v == 'False')
        return v.lower() in ('true')
    
    def none_or_str(value):
        if value.lower() == 'none':
            return None
        return value

    parser = argparse.ArgumentParser(description='Parameter settings for end-to-end optimization')

    # Main objective
    parser.add_argument('--mode',       type=str, choices=['Original', 'Normalize'], default='Original')
    
    # WandB & Sweep configurations
    parser.add_argument('--wandb',      type=str2bool, default=False)
    parser.add_argument('--sweep',      type=str2bool, default=False)
    parser.add_argument('--project',    type=str, default='Project')
    parser.add_argument('--sweep_name', type=str, default='Sweep')
    
    # Foward network arguments
    parser.add_argument('--fwd_path' ,          type=str, required=True, help='Path of the forward net')
    parser.add_argument('--model_type' ,        type=str, choices=['LSTM', 'Conv1D', 'FCN'], required=True, help='Path of the forward net')
    parser.add_argument('--data_dir' ,          type=str, default='./fwdnet/gen_20240611/preprocessed/', help='Path of the training dataset')
    parser.add_argument('--nk_norm_method',     type=str, choices=['z', 'linear'], required=True, help='Normalization method of nk')
    parser.add_argument('--geom_norm_method',   type=str, choices=['z', 'linear'], required=True, help='Normalization method of geom')
    
    # Loss arguments
    parser.add_argument('--batch_weights'       , type=str, default='1.0,1.0,1.0')
    parser.add_argument('--BCE_weight'          , type=float, default=10.0)
    parser.add_argument('--Match_weight'        , type=float, default=1000.0)
    parser.add_argument('--Unmatch_weight'      , type=float, default=0.0002)
    parser.add_argument('--Eff_weight'          , type=float, default=60.0)
    parser.add_argument('--Norm_weight'         , type=float, default=100.0)
    parser.add_argument('--P_loss_weight'       , type=float, default=0.0)
    parser.add_argument('--Spatial_loss_weight' , type=float, default=0.0)
    
    # Training arguments
    parser.add_argument('--steps'       , type=int, default=1000, help='Total number of optimization cycles')
    parser.add_argument('--step_term'   , type=int, default=10, help='Total number of optimization cycles')
    
    # Metasurface arguments
    parser.add_argument('--init',               type=str, choices=['random', 'preoptimized'], help='Metasurface initialization')
    parser.add_argument('--path_holo',          type=str, default='./gsalgorithm_results/meta_dict_nk_sinx_N2_23_h=810_p=400')
    parser.add_argument('--num_meta',           type=int, default=500, required=True, help='Number of meta-atoms on an axis')
    parser.add_argument('--target_wavelengths', type=str, default="450.0e-9,532.0e-9,635.0e-9", help='Target wavelengths')
 
    # Optimization arguments
    parser.add_argument('--Mat_lr',     type=float, default=1e-3)
    parser.add_argument('--HP_lr',      type=float, default=1e-3)
    parser.add_argument('--LW_lr',      type=float, default=1e-1)
    parser.add_argument('--NumP_lr',    type=float, default=1e-3)
    parser.add_argument('--Rot_lr',     type=float, default=1e-3)
    parser.add_argument('--Mhp_beta1',  type=float, default=0.9)
    parser.add_argument('--Lwnr_beta1', type=float, default=0.9)
    parser.add_argument('--ar_lim',     type=float, default=10.0)

    args = parser.parse_args()
    args.target_wavelengths = [float(w) for w in args.target_wavelengths.split(',')]
    args.batch_weights = [float(bw) for bw in args.batch_weights.split(',')]
    
    if args.model_type == "LSTM":
        fwdnet_file = args.fwd_path.split('/')[-2]
        init = fwdnet_file.split('init_')[1][:3]
        if fwdnet_file.find('hidden_size') != -1:
            hs = int(fwdnet_file.split('hidden_size_')[1].split('_nlayer')[0])
            channels = [int(i) for i in fwdnet_file.split('fc_feature_layer_')[1].split('_hidden_size')[0].split('_')]
        else:
            hs = int(fwdnet_file.split('hid_size_')[1].split('_nlayer')[0])
            channels = [int(i) for i in fwdnet_file.split('fc_feature_layer_')[1].split('_hid_size')[0].split('_')]
        nl = int(fwdnet_file.split('_nlayer_')[1])
        fc_layers = [{'in_channels': channels[i], 'out_channels': channels[i+1]} for i in range(len(channels)-1)]
        addition_args = {'init': init,
                         'input_size': 7,
                         'hidden_size': hs,
                         'num_layer': nl,
                         'fc_feature_layer': fc_layers}

    elif args.model_type == "FCN":
        addition_args = {'fc_feature_layer' : [{'in_channels' : 6, 'out_channels' : 200},
                                               {'in_channels' : 200, 'out_channels' : 200},
                                               {'in_channels' : 200, 'out_channels' : 200},
                                               {'in_channels' : 200, 'out_channels' : 200},
                                               {'in_channels' : 200, 'out_channels' : 200},
                                               {'in_channels' : 200, 'out_channels' : 200},
                                               {'in_channels' : 200, 'out_channels' : 200},
                                               ]}

    elif args.model_type == "ConvT1D" :
        addition_args = {'conv_feature_layer' : [{'in_channels' : 6, 'out_channels' : 32, 'kernerl_size' : 2, 'stride' : 2, 'padding' : 1},  # (num_features, 6) -> (32,4)
                                        {'in_channels' : 32, 'out_channels' : 64, 'kernerl_size' : 2, 'stride' : 2, 'padding' : 1},  # (32,4) -> (64,3)
                                        {'in_channels' : 64, 'out_channels' : 128, 'kernerl_size' : 2, 'stride' : 1, 'padding' : 0},  # (64,3) -> (128,2)
                                        ],
                        'fc_feature_layer' : [{'in_channels' : 256, 'out_channels' : 512}, # first in_channels should be H*C of the last convlayer ouput (128,2)
                                        {'in_channels' : 512, 'out_channels' : 256},
                                        {'in_channels' : 256, 'out_channels' : 256},
                                        {'in_channels' : 256, 'out_channels' : 256},
                                        {'in_channels' : 256, 'out_channels' : 256},
                                        {'in_channels' : 256, 'out_channels' : 64},
                                        ]}
 
    args.additional_args = addition_args
    
    print(args)
    return args