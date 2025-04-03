import torch
import utils.toolkits as toolkits
import wandb
from tqdm import tqdm
import conv
import loss
import os 
import numpy as np

# @record
def optimize(args) -> None:    
    # Initialize params
    params = toolkits.initialize_params(args)
    
    # Check configurations
    iswandb = params['wandb']
    total_step = params['steps']
    step_term = params['step_term']
    device = params['device']
    seed = params['seed']
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    wdb_idx = np.random.randint(0, params['num_meta'] ** 2)
        
    # Load fwd network
    _net = toolkits.load_fwd(params, device)
    net = torch.nn.DataParallel(_net).to(device)
    net.eval()
    
    # Initial variables
    mat_initial, H_initial, P_initial, LW_initial, numP_initial, rot_initial = toolkits.init_mat_geom(params)
    Mat_var = mat_initial.to(device)
    H_var = H_initial.to(device)
    P_var = P_initial.to(device)
    LW_var = LW_initial.to(device)
    NumP_var = numP_initial.to(device)
    Rot_var = rot_initial.to(device)
        
    torch.backends.cudnn.enabled = False # --> Allocates too much GPU memory
    
   # Optimizers
    Optimizer_mhp = torch.optim.Adam([{'params': Mat_var, 'lr': params['mat_lr']},
                                      {'params': H_var,  'lr': params['h_lr']},
                                      {'params': P_var,  'lr': params['p_lr']},
                                      ], betas=(0.9, 0.99))
    Optimizer_lwnr = torch.optim.Adam([{'params': LW_var,  'lr': params['lw_lr']},
                                       {'params': NumP_var,'lr': params['numP_lr']},
                                       {'params': Rot_var, 'lr': params['rot_lr']},
                                       ], betas=(0.8, 0.9))
    
    scheduler_mhp = torch.optim.lr_scheduler.LambdaLR(Optimizer_mhp, lr_lambda = lambda epoch: 0.98 ** epoch)
    # scheduler_mhp = torch.optim.lr_scheduler.LambdaLR(Optimizer_mhp, lr_lambda = lambda epoch: 0.995 ** epoch)
    
    # Set loss function
    hloss_function = loss.loss_functions(params)
    
    with torch.no_grad():
        rot_2d = toolkits.rot_to_2d(Rot_var, params)
        fwd_input = toolkits.create_net_input(Mat_var, H_var, P_var, LW_var, NumP_var, params)
        tm_2d, tp_2d = toolkits.run_extract(fwd_input, net, params)
        metasurface_mask = conv.define_metasurface(tm_2d, tp_2d, rot_2d)
        I = conv.compute_hologram(metasurface_mask)
        _, loss_components = hloss_function(I, tm_2d)
    
    for step in tqdm(range(total_step)):
        # Logging
        if step % step_term == 0:
            # HP
            h_unnorm = toolkits.norm_or_restore(params['data_dir'], H_var.clone().detach(), mode='restore', type='h')
            p_unnorm = toolkits.norm_or_restore(params['data_dir'], P_var.clone().detach(), mode='restore', type='p')
               
            log_H = h_unnorm
            log_P = p_unnorm
            log_LW = LW_var.clone().detach()
            log_LW = LW_var.clone().detach()
            log_numP = torch.argmax(NumP_var.clone().detach(), dim=-1) + 1
            log_rot = torch.argmax(Rot_var.clone().detach(), dim=-1)
                
            if iswandb:
                wandb_log = {}
                I_for_wandb = I.detach().cpu().clone()
                I_for_wandb = I_for_wandb / torch.max(torch.max(I_for_wandb, dim=0, keepdims=True)[0], dim=1, keepdims=True)[0]
                # Exclude the center pixels
                num = I.shape[1] 
                idx_center = [i for i in range(num) if (abs(i-num//2) < 2)]
                I_for_wandb[idx_center[0] : idx_center[-1]+1, idx_center[0]: idx_center[-1]+1, :] = 0
                I_norm = I_for_wandb * 255
                I_B = np.zeros([params['num_meta'], params['num_meta'], 3]); I_G = I_B.copy(); I_R = I_B.copy()
                I_B[:, :, 2] = np.array(I_norm[:, :, 0]); I_G[:, :, 1] = np.array(I_norm[:,:,1]); I_R[:, :, 0] = np.array(I_norm[:,:,2])                    
                I_total = np.array(I_norm)
                for i in range(3):
                    I_total[:, :, i] = 255 * I_total[:, :, 2-i] / np.max(I_total[:, :, 2-i])
                wandb_log[f'Material'] = torch.argmax(Mat_var.clone().detach()).item()
                wandb_log[f'H'] = log_H.item()
                wandb_log[f'P'] = log_P.item()
                wandb_log[f'L_{wdb_idx}'] = log_LW[wdb_idx, 0].item()
                wandb_log[f'W_{wdb_idx}'] = log_LW[wdb_idx, 1].item()
                wandb_log[f'NumP_{wdb_idx}'] = log_numP[wdb_idx].item()
                wandb_log[f'Rot_{wdb_idx}'] = log_rot[wdb_idx].item()     
                wandb_log[f'Image_R'] = wandb.Image(I_R)
                wandb_log[f'Image_G'] = wandb.Image(I_G)
                wandb_log[f'Image_B'] = wandb.Image(I_B)
                wandb_log[f'Image_total'] = wandb.Image(I_total)
                wandb_log[f'TM_B '] = torch.mean(tm_2d[:, :, 0]).item()
                wandb_log[f'TM_G '] = torch.mean(tm_2d[:, :, 1]).item()
                wandb_log[f'TM_R '] = torch.mean(tm_2d[:, :, 2]).item()
                # wandb_log[f'ratio_ROI_B'] = torch.sum(I_for_wandb[270:720, 500:-500, 0]) / torch.sum(I_for_wandb[:, :, 0])
                # wandb_log[f'ratio_ROI_G'] = torch.sum(I_for_wandb[270:720, 500:-500, 1]) / torch.sum(I_for_wandb[:, :, 1])
                # wandb_log[f'ratio_ROI_R'] = torch.sum(I_for_wandb[270:720, 500:-500, 2]) / torch.sum(I_for_wandb[:, :, 2])
                wandb_log[f'ratio_ROI_B'] = torch.sum(I_for_wandb[180:480, 340:-340, 0]) / torch.sum(I_for_wandb[:, :, 0])
                wandb_log[f'ratio_ROI_G'] = torch.sum(I_for_wandb[180:480, 340:-340, 1]) / torch.sum(I_for_wandb[:, :, 1])
                wandb_log[f'ratio_ROI_R'] = torch.sum(I_for_wandb[180:480, 340:-340, 2]) / torch.sum(I_for_wandb[:, :, 2])
                wandb_log_loss = {f'{key}_loss': val for [key, val] in loss_components.items()}
                wandb_log_loss['Total_loss'] = sum(loss_components.values())
                wandb_log = dict(wandb_log, **wandb_log_loss) 
                wandb.log(wandb_log)
    
            if (not params['sweep']) or not('sweep' in params.keys()):
                dir_trn_result = f"./project_logs/{params['project']}_result"
                os.makedirs(dir_trn_result, exist_ok=True)
                load_dict = {}
                load_dict[f'nk_idx'] = torch.argmax(Mat_var.clone().detach()).item()
                load_dict[f'H'] = log_H.item()
                load_dict[f'P'] = log_P.item()
                load_dict[f'LW'] = LW_var.clone().detach().cpu()
                load_dict[f'NumP'] = torch.argmax(NumP_var.clone().detach().cpu(), dim=-1)
                load_dict[f'Rot'] = torch.argmax(Rot_var.clone().detach().cpu(), dim=-1)
                load_dict[f'Image'] = I.clone().detach().cpu()
                torch.save(load_dict, dir_trn_result + f"/{step}.pt")
        
        # Material, height, pitch optimization
        Mat_var.requires_grad_(True)
        H_var.requires_grad_(True)
        P_var.requires_grad_(True)
        LW_var.requires_grad_(False)
        NumP_var.requires_grad_(False)
        Rot_var.requires_grad_(False)
        rot_2d = toolkits.rot_to_2d(Rot_var, params)
        for _ in range(4):
            Optimizer_mhp.zero_grad()
            fwd_input = toolkits.create_net_input(Mat_var, H_var, P_var, LW_var, NumP_var, params)
            tm_2d, tp_2d = toolkits.run_extract(fwd_input, net, params)
            metasurface_mask = conv.define_metasurface(tm_2d, tp_2d, rot_2d)
            I = conv.compute_hologram(metasurface_mask)
            loss_mhp, loss_components = hloss_function(I, tm_2d)
            loss_mhp.backward()
            Optimizer_mhp.step()
            scheduler_mhp.step()
            # Keeping the variable within range
            Mat_var, H_var, P_var = toolkits.var_clamp('MHP', params, h=H_var, p=P_var, mat=Mat_var)

        # Length, width, number of pillars, and rotation optimization
        Mat_var.requires_grad_(False)
        H_var.requires_grad_(False)
        P_var.requires_grad_(False)
        LW_var.requires_grad_(True)
        NumP_var.requires_grad_(True)
        Rot_var.requires_grad_(True)
        for _ in range(10):        
            Optimizer_lwnr.zero_grad()
            hook_lw = LW_var.register_hook(lambda grad: 1e3 *grad)
            hook_numP   = NumP_var.register_hook(lambda grad: 1e4 * grad)
            hook_rot    = Rot_var.register_hook(lambda grad: 1e4 * grad)
            fwd_input = toolkits.create_net_input(Mat_var, H_var, P_var, LW_var, NumP_var, params)
            tm_2d, tp_2d = toolkits.run_extract(fwd_input, net, params)
            rot_2d = toolkits.rot_to_2d(Rot_var, params)      
            metasurface_mask = conv.define_metasurface(tm_2d, tp_2d, rot_2d)
            I = conv.compute_hologram(metasurface_mask)
            loss_lwnr, loss_components = hloss_function(I, tm_2d)
            loss_lwnr.backward()
            torch.nn.utils.clip_grad_norm_([LW_var, NumP_var, Rot_var], max_norm=1.0)
            Optimizer_lwnr.step() 
            hook_lw.remove()
            hook_numP.remove()
            hook_rot.remove()
            # Keeping the variable within range
            LW_var, NumP_var, Rot_var  = toolkits.var_clamp('LWNR', params, h=H_var, p=P_var, lw=LW_var, numP=NumP_var, rot=Rot_var)

        # Clear cache to free memory
        torch.cuda.empty_cache()
        