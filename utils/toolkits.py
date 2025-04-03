import torch
import numpy as np
import os
import math
from fwdnet.jbnet import *
import cv2
from utils.lens_distort import lensdistort

def initialize_params(args):
    params = dict(**args)
    
    # Units and tensor dimensions.
    params['nanometers'] = 1E-9
    
    # Nk list
    nk_dir = params['nk_dir']
    file_list = os.listdir(nk_dir)
    file_list.sort()
    params['nk_f_list'] = file_list
    
    # Set device
    device = torch.device("cuda")
    params['device'] = device
    
    # Set target image
    target_images, list_mask, list_reverse_mask = gen_target_images(params)
    params['target_images'] = target_images.to(device)
    params['list_mask'] = list_mask
    params['list_reverse_mask'] = list_reverse_mask
    
    return params


# # Scaling & Distortion
# def gen_target_images(params) -> None:
#     img_dir = '/home/linkle115/vscode_server/EEasih/source_image/'
#     type_img = params['target_img']
#     original_img = [type_img + '_' + c + '.png' for c in ['Blue', 'Green', 'Red']]
#     wl = params['wl']
#     wl_standard = min(wl)
#     I = np.zeros([params['num_meta'], params['num_meta']], 3)
#     for i in range(3):
#         wl_i = wl[i]
#         f_img = original_img[i]
#         I_source = cv2.imread(img_dir + f_img, cv2.IMREAD_GRAYSCALE)
        
#         # Scaling
#         size1 = int(params('num_meta') * wl_standard / wl_i)
#         I_source = cv2.resize(I, (size1, size1))
#         I_source = I_source / np.max(I)
#         # Bilateral symmetric conversion
#         # In most cases, light is incident from substrate to metasurface, so we need this conversion. If not, ignore this.
#         I_source = I_source[:, ::-1]
#         p = params['pitch']
#         def idxtocartesian(idx, pitch, nn): return pitch * (1/2 + idx - nn/2)
#         def cartesiantoidx(x, pitch, nn): return int(nn/2 + x/pitch - 1/2)
#         def distort(idx0, idx1, lam, pitch, D):
#             x = idxtocartesian(idx1, pitch, params['num_meta'])
#             y = idxtocartesian(idx0, pitch, params['num_meta'])
#             x_ = x / math.sqrt(1 + (x**2 + y**2) * (lam / (D * pitch))**2)
#             y_ = y / math.sqrt(1 + (x**2 + y**2) * (lam / (D * pitch))**2)
#             idx0_ = cartesiantoidx(y_, pitch, params['num_meta'])
#             idx1_ = cartesiantoidx(x_, pitch, params['num_meta'])
#             return idx0_, idx1_
            
#         # Distortion correction
#         for idx1 in range(size1):
#             for idx2 in range(size1):
#                 idx1_new, idx2_new = distort(idx1, idx2, wl_i, wl_standard, 1)
#                 I[idx1_new, idx2_new, i] = I_source[idx1, idx2]
    
#     target_img = torch.from_numpy(I).to(device=params['device'], dtype=torch.float32)
#     crit = 0.3
#     list_mask = []
#     list_reverse_mask = []
#     for i in range(3):
#         list_mask.append(torch.where(target_img[:, :, i] > crit, 1.0, 0))
#         prev_idx = (i - 1) % 3
#         next_idx = (i + 1) % 3
#         list_reverse_mask.append(torch.where((target_img[:, :, i] < crit) & ((target_img[:, :, prev_idx] > crit) | (target_img[:, :, next_idx] > crit)), 1.0, 0))
#     return target_img, list_mask, list_reverse_mask

def gen_target_images(params) -> None:
    img_dir = '/home/linkle115/vscode_server/EEasih/source_image/'
    type_img = params['target_img']
    original_img = [type_img + '_' + c + '.png' for c in ['Blue', 'Green', 'Red']]
    img = np.zeros([params['num_meta'], params['num_meta'], 3])
    for i in range(3):
        f_img = original_img[i]
        I = cv2.imread(img_dir + f_img, cv2.IMREAD_GRAYSCALE)
        I = cv2.resize(I, (params['num_meta'], params['num_meta']))
        I = I / np.max(I)
        # Bilateral symmetric conversion
        # In most cases, light is incident from substrate to metasurface, so we need this conversion. If not, ignore this.
        I = I[:, ::-1]
        
        clev = 0.6
        white_edge = 20
        I = lensdistort(I,clev, ftype=3, bordertype='fit')    # "lensdistort" is a help function
        for idx1 in range(params['num_meta']):
            for idx2 in range(params['num_meta']):
                if idx1 < white_edge or idx1 > (params['num_meta']-white_edge) or idx2 < white_edge or idx2 > (params['num_meta']-white_edge):   # boundaries can be changed after checking "compensated_img"
                    I[idx1, idx2] = 0                 # removal of white edges
        I[I<0] = 0                              # removal of negative values
        I = I/np.max(I)                         # renormalization
        img[:, :, i] = I            # BGR to RGB
    
    target_img = torch.from_numpy(img).to(device=params['device'], dtype=torch.float32)
    crit = 0.3
    list_mask = []
    list_reverse_mask = []
    for i in range(3):
        list_mask.append(torch.where(target_img[:, :, i] > crit, 1.0, 0))
        prev_idx = (i - 1) % 3
        next_idx = (i + 1) % 3
        list_reverse_mask.append(torch.where((target_img[:, :, i] < crit) & ((target_img[:, :, prev_idx] > crit) | (target_img[:, :, next_idx] > crit)), 1.0, 0))
    return target_img, list_mask, list_reverse_mask


def load_nk(device, nk_dir, nk_file_list):
    num_nk = len(nk_file_list)
    nk_total = np.zeros([num_nk, 31, 2])
    for i in range(num_nk):
        nk_total[i] = torch.from_numpy(np.load(nk_dir + nk_file_list[i]))
    return torch.from_numpy(nk_total).to(dtype=torch.float32, device=device)


def load_GS_results(params):
    import pickle
    path_holo = params['path_holo']
    with open(path_holo, 'rb') as f:
        fname = path_holo.split('/')[-1]
        dict_meta = pickle.load(f)
    mat_name = fname.split('nk_')[1].split('h')[0] + '.npy'
    file_list = params['nk_f_list']
    mat_idx = file_list.index(mat_name)
    h = np.array(int(fname.split('h=')[1][:3])); p=np.array(int(fname.split('p=')[1][:3]))
    lwn = dict_meta['LWN']
    unitcell_indicator = dict_meta['unitcell_indicator']
    rotation_indicator = dict_meta['rotation_indicator']
    lwn_2d = lwn[unitcell_indicator]
    lwn_flat = lwn_2d.reshape(-1, 3)
    lw_flat = np.expand_dims(np.int32(lwn_flat[:, 0:2]), 1)
    numP_flat = np.expand_dims(np.int32(lwn_flat[:, 2]), (1, 2))
    rot_flat = rotation_indicator.reshape(-1, 1)
    return mat_idx, h, p, lw_flat, numP_flat, rot_flat


def init_mat_geom(params):
    np.random.seed(params['seed'])
    def generate_w_and_l(g_in, h_in, p_in, num_in, AR_lim=10):
        lw_min = max((h_in / AR_lim) // 10 * 10, 40) * np.ones_like(num_in)
        w_max = np.min([(np.sqrt(p_in ** 2 - lw_min ** 2) - g_in) / num_in, p_in / math.sqrt(2) * np.ones_like(num_in)])
        w = 10 * np.random.randint(lw_min // 10, w_max // 10)
        mask_1 = (num_in == 1)
        l_min_1 = 10 * np.max([lw_min // 10 * np.ones_like(w), w // 10 + 1], axis=0)
        l_min_not1 = lw_min
        l_1 = 10 * np.random.randint(l_min_1 // 10, np.sqrt(p_in ** 2 - w ** 2)  // 10 + 1)
        l_not1 = 10 * np.random.randint(l_min_not1 // 10, np.sqrt(p_in ** 2 - (num_in * w + g_in) ** 2) // 10 + 1)
        l = mask_1 * l_1 + (1-mask_1) * l_not1
        return w, l
    
    def scale_to_one(mat):
        mat_sum = torch.sum(mat, dim=-1)
        return mat / mat_sum.unsqueeze(-1)
    
    num_X = params["num_meta"]
    num_Y = num_X
    
    # Start with numP = 1
    numP_var = 0.3 * np.random.rand(num_X * num_Y, 3)
    numP_var[:, 0] = 0.4
    # numP_var = np.random.rand(batch_size, num_X * num_Y, 3)
    rot_var  = np.random.rand(num_X * num_Y, 6)
    if params['init'] == 'random':
        mat_var = np.random.rand(len(params['nk_f_list']))
        h_var = np.random.randint(60, 90+1, [1,]) * 10
        p_var = np.random.randint(38, 40+1, [1,]) * 10
        w_arr = np.zeros([num_X * num_Y]); l_arr = np.zeros([num_X * num_Y])
        numP = np.argmax(numP_var, axis=-1, keepdims=True) + 1
        g_in = np.where(numP == 1, 0, np.where(numP == 2, 50, 100))
        w_arr, l_arr = generate_w_and_l(g_in, h_var, p_var, numP, params['ar_lim'])
        lw_var = np.concatenate([l_arr, w_arr], axis=-1)  # size: [numX * numY, 1, 2]

    elif params['init'] == 'preoptimized':
        mat_idx, h_var, p_var, lw_var, int_numP, int_rot = load_GS_results(params)
        # Mat
        mat_var = 0.6 * np.random.rand(len(params['nk_f_list']))
        mat_var[mat_idx] *= 0.8
        
        # NumP & Rot
        numP_var *= 0.3
        numP_var[np.arange(int_numP.shape[0]), int_numP[0, 0] - 1] = 0.4
        rot_var *= 0.2
        rot_var[np.arange(int_rot.shape[0]), int_rot[0]] = 0.3
        
    # Numpy to torch
    mat_var = torch.from_numpy(mat_var).to(torch.float32)
    h_var = torch.from_numpy(h_var).to(torch.float32)
    p_var = torch.from_numpy(p_var).to(torch.float32)
    lw_var = torch.from_numpy(lw_var).to(torch.float32)
    numP_var = torch.from_numpy(numP_var).to(torch.float32)
    rot_var = torch.from_numpy(rot_var).to(torch.float32)
    
    mat_norm = scale_to_one(mat_var)
    h_norm = norm_or_restore(params['data_dir'], h_var, mode='normalize', type='h')
    p_norm = norm_or_restore(params['data_dir'], p_var, mode='normalize', type='p')
    lw_norm = norm_or_restore(params['data_dir'], lw_var, mode='normalize', type='lw')
    numP_norm = scale_to_one(numP_var)
    rot_norm = scale_to_one(rot_var)
        
    return mat_norm, h_norm, p_norm, lw_norm, numP_norm, rot_norm


def differentiable_argmax(x, type):
    if type == 'numP':
        mat = torch.tensor([[1], [2], [3]]).to(dtype=torch.float32, device=x.device)
    elif type == 'rot':
        mat = torch.tensor([[1], [2], [3], [4], [5], [6]]).to(dtype=torch.float32, device=x.device)
    soft_selection = torch.nn.functional.softmax(x, dim=-1)
    selected_idx = soft_selection.argmax(dim=-1)
    hard_selection = torch.zeros_like(x)
    hard_selection[[i for i in range(selected_idx.shape[0])], selected_idx] = 1
    hard_selection = hard_selection - soft_selection.detach() + soft_selection
    x_selected = torch.matmul(hard_selection, mat)
    return x_selected

def norm_or_restore(data_dir, var, mode, type, **kwargs):
    if type == 'NK':
        mean = torch.load(data_dir + 'NK_mean.pt').to(dtype=torch.float32)
        std = torch.load(data_dir + 'NK_std.pt').to(dtype=torch.float32)
    else:
        mean = torch.load(data_dir + 'Geom_mean.pt').to(dtype=torch.float32)
        std = torch.load(data_dir + 'Geom_std.pt').to(dtype=torch.float32)
        if type == 'h':
            idx = 1
        elif type == 'p':
            idx = 2
        elif type == 'l':
            idx = 3
        elif type == 'w':
            idx = 4
        elif type == 'lw':
            idx = [3, 4]
        elif type == 'numP':
            idx = 0
        mean = mean[idx]
        std = std[idx] + 1e-10
    
    if 'device' in kwargs.keys():
        device = kwargs['device']
        mean = mean.to(device); std = std.to(device)
    
    if mode == 'normalize':
        var_restored = (var - mean) / std
    elif mode == 'restore':
        var_restored = var * std + mean
    
    return var_restored


def create_net_input(mat, h_norm, p_norm, lw_norm, numP, params):
    device = params['device']
    data_dir = params['data_dir']
    num_total = params['num_meta'] ** 2
    
    soft_selection  = torch.nn.functional.softmax(mat, dim=-1)
    selected_idx = soft_selection.argmax(dim=-1)
    hard_selection = torch.zeros_like(mat)
    hard_selection[selected_idx] = 1.0
    hard_selection = hard_selection - soft_selection.detach() + soft_selection
    
    # NK
    nk_total = load_nk(device, params['nk_dir'], params['nk_f_list'])    
    nk_selected = torch.matmul(nk_total.permute(1,2,0), hard_selection) # [Num_total, 31, 2]
    nk_elong = torch.tile(nk_selected.unsqueeze(0), dims=(num_total, 1, 1)) 
    nk_elong_norm = norm_or_restore(data_dir, nk_elong, mode='normalize', type='NK', device=device)
    
    # NumP
    numP_ = differentiable_argmax(numP, 'numP')
    numP_norm = norm_or_restore(data_dir, numP_, mode='normalize', type='numP', device=device)
    
    # H, P
    h_unnorm = norm_or_restore(data_dir, h_norm, mode='restore', type='h', device=device)
    p_unnorm = norm_or_restore(data_dir, p_norm, mode='restore', type='p', device=device)
    with torch.no_grad():
        h_unnorm.round_(decimals=-1); p_unnorm.round_(decimals=-1)
    h_norm_rounded = norm_or_restore(data_dir, h_unnorm, mode='normalize', type='h', device=device)
    p_norm_rounded = norm_or_restore(data_dir, p_unnorm, mode='normalize', type='p', device=device)
    
    # LW
    lw_unnorm = norm_or_restore(data_dir, lw_norm, mode='restore', type='lw', device=device)
    with torch.no_grad():
        lw_unnorm.round_(decimals=-1)
    lw_norm_rounded = norm_or_restore(data_dir, lw_unnorm, mode='normalize', type='lw', device=device)
                
    numP_elong = torch.tile(numP_norm.unsqueeze(-1), dims=(1, 31, 1))
    h_elong = torch.tile(h_norm_rounded.view(-1, 1, 1), dims=(num_total, 31, 1))
    p_elong = torch.tile(p_norm_rounded.view(-1, 1, 1), dims=(num_total, 31, 1))
    lw_elong = torch.tile(lw_norm_rounded.unsqueeze(-2), dims=(1, 31, 1))
    geom_elong = torch.concat([numP_elong, h_elong, p_elong, lw_elong], dim=-1)
    return torch.concat([nk_elong_norm, geom_elong], dim=-1)


def load_fwd(params, device):
    checkpoint = torch.load(params['fwd_path'])    
    fwd_model = params['fwd_model']
    fwd_args = params['additional_args']
    if fwd_model == 'LSTM':
        net = LSTM(**fwd_args, device=device)
        if params['fwd_path'].find("TxyR") == -1:
            net.build_fc_layer(input_dim=7, output_dim=2)
        else:
            net.build_fc_layer(input_dim=7, output_dim=3)
    
    net.load_state_dict(checkpoint['network'])
    return net


class UniqueWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        # Use numpy to find unique rows, but we'll also need to track which rows were selected
        unique, inverse_indices = torch.unique(A, dim=0, return_inverse=True)
        ctx.save_for_backward(inverse_indices, torch.tensor(A.shape, device=A.device))
        return unique, inverse_indices
    @staticmethod
    def backward(ctx, grad_unique, grad_inverse_indices):
        inverse_indices, original_shape = ctx.saved_tensors
                
        # Expand grad_unique to match the dimensions of the input tensor
        grad_expanded = grad_unique[inverse_indices]

        return grad_expanded, None

def unique_with_grad(A):
    return UniqueWithGrad.apply(A)

def run_extract(fwd_input, net, params):
    num_X = params['num_meta']
    def c_intp_mag(x1, x2, x_mid, y1, y2):
        return y1 + (y2 - y1) / (x2 - x1) * (x_mid - x1)
    def c_intp_phase(x1, x2, x_mid, ar_y1, ar_y2):
        y_mid = torch.zeros_like(ar_y1, device=params['device'])
        for i, (y1, y2) in enumerate(zip(ar_y1, ar_y2)):
            if y1 * y2 >= 0:
                y_mid[i] = y1 + (y2 - y1) / (x2 - x1) * (x_mid - x1)
            elif y1 < y2: # y1 < 0 < y2
                y_minus = y1 + 2; y_plus = y2; x_minus = x1; x_plus = x2
                y_mid[i] =  y_minus + (y_plus - y_minus) / (x_plus - x_minus) * (x_mid - x_minus)
            elif y2 < y1: # y2 < 0 < y1
                y_minus = y2 + 2; y_plus = y1; x_minus = x2; x_plus = x1
                y_mid[i] =  y_minus + (y_plus - y_minus) / (x_plus - x_minus) * (x_mid - x_minus)
        y_mid[y_mid > 1] -= 2
        return y_mid 
    
    def txy_to_tmp(tx, ty):
        tm = tx ** 2 + ty ** 2
        epsilon = 1e-10
        safe_tx = tx + epsilon * torch.sign(tx)
        tp = torch.atan(ty / (safe_tx)) / torch.pi \
        - torch.logical_and(tx < 0, ty < 0).type(torch.float32) \
        + torch.logical_and(tx < 0, ty > 0).type(torch.float32)
        return tm, tp
    
    wl = [round(float(i) / params['nanometers']) for i in params['wl']]
    num_wl = len(wl)

    fwd_input_reshaped = fwd_input.view(-1, fwd_input.shape[-2], fwd_input.shape[-1])
    u_fwd_input, inverse_indices = unique_with_grad(fwd_input_reshaped)
    u_fwd_output = net(u_fwd_input)
    u_tm = torch.zeros([u_fwd_output.shape[0], num_wl], device=params['device'])
    u_tp = torch.zeros([u_fwd_output.shape[0], num_wl], device=params['device'])
    for i in range(num_wl):
        x_i1 = (wl[i] // 10) * 10; x_i2 = x_i1 + 10
        idx_i1 = (x_i1 - 400) // 10; idx_i2 = idx_i1 + 1
        tm_i1, tp_i1 = txy_to_tmp(u_fwd_output[:, idx_i1, 0], u_fwd_output[:, idx_i1, 1])
        tm_i2, tp_i2 = txy_to_tmp(u_fwd_output[:, idx_i2, 0], u_fwd_output[:, idx_i2, 1])
        u_tm[:, i] = c_intp_mag(x_i1, x_i2, wl[i], tm_i1, tm_i2)
        u_tp[:, i] = c_intp_phase(x_i1, x_i2, wl[i], tp_i1, tp_i2)
    
    # Map u_tm to tm
    tm = u_tm[inverse_indices]
    tp = u_tp[inverse_indices]
    tm_reshaped = tm.view(num_X, num_X, num_wl)
    tp_reshaped = tp.view(num_X, num_X, num_wl)

    return tm_reshaped, tp_reshaped


def rot_to_2d(rot, params):
    num_X = params['num_meta']
    rot_reshaped = differentiable_argmax(rot, 'rot').reshape(num_X, num_X, 1)
    return rot_reshaped


def var_clamp(var, params, **kwargs):
    data_dir = params['data_dir']
    # # H, P is always used
    # p = kwargs['p']; h = kwargs['h']
    # p_unnorm = p.clone().detach()
    # h_unnorm = h.clone().detach()
        
    if var == 'MHP':
        mat = kwargs['mat']; p = kwargs['p']; h = kwargs['h'] 
        with torch.no_grad():
            # Mat
            mat_ = mat.clone().detach()
            mat_sum = torch.sum(mat_, dim=-1)
            mat_norm = mat / mat_sum.unsqueeze(-1)
            mat.data = mat_norm
            # H, P
            h_unnorm = norm_or_restore(data_dir, h.clone().detach(), mode='restore', type='h')
            p_unnorm = norm_or_restore(data_dir, p.clone().detach(), mode='restore', type='p')
            h_unnorm.clamp_(600, 900); p_unnorm.clamp_(380, 400)
            h_norm = norm_or_restore(data_dir, h_unnorm, mode='normalize', type='h', device=params['device'])
            p_norm = norm_or_restore(data_dir, p_unnorm, mode='normalize', type='p', device=params['device'])
            h.data = h_norm; p.data = p_norm
        return mat, h, p
    
    elif var == 'LWNR':
        h = kwargs['h']; p = kwargs['p']; lw = kwargs['lw']; numP = kwargs['numP']; rot=kwargs['rot']
        with torch.no_grad():
            # NumP
            numP_ = numP.clone().detach()
            numP_norm = numP_ / (torch.sum(numP_, dim=-1).unsqueeze(-1) + 1e-10)
            numP.data = numP_norm
            
            num = torch.argmax(numP.clone().detach(), dim=-1, keepdim=True).squeeze(-1) + 1
            idx_1_num = torch.where(num == 1)[0]
            idx_2_num = torch.where(num == 2)[0]
            idx_3_num = torch.where(num == 3)[0]
            h_unnorm = norm_or_restore(data_dir, h.clone().detach(), mode='restore', type='h')
            h_per_ar = h_unnorm / params['ar_lim']
            p_unnorm = norm_or_restore(data_dir, p.clone().detach(), mode='restore', type='p')
            ## W
            w_min = torch.ones_like(num) * norm_or_restore(data_dir, h_per_ar.clone(),  mode='normalize', type='w')
            w_max = 1.0 * torch.zeros_like(num)
            # w_max[idx_1_num] = norm_or_restore(data_dir, torch.tensor([260.1], device=params['device']), mode='normalize', type='w')
            # w_max[idx_2_num] = norm_or_restore(data_dir, torch.tensor([150.1], device=params['device']), mode='normalize', type='w')
            # w_max[idx_3_num] = norm_or_restore(data_dir, torch.tensor([80.1], device=params['device']),  mode='normalize', type='w')
            w_max[idx_1_num] = norm_or_restore(data_dir, torch.tensor([p_unnorm / math.sqrt(2) - 10.0], device=params['device']), mode='normalize', type='w')
            # w_max[idx_1_num] = norm_or_restore(data_dir, torch.tensor([p_unnorm / torch.sqrt(2) - 10.0], device=params['device']), mode='normalize', type='w')
            w_max[idx_2_num] = norm_or_restore(data_dir, torch.tensor([(p_unnorm - 100) / 2 + 0.1], device=params['device']), mode='normalize', type='w')
            w_max[idx_3_num] = norm_or_restore(data_dir, torch.tensor([(p_unnorm - 150) / 3 + 0.1], device=params['device']),  mode='normalize', type='w')
            lw[:, 1].clamp_(min=w_min, max=w_max)                        
            
            ## L 
            # l_min = torch.ones_like(num) * norm_or_restore(data_dir, h_per_ar,  mode='normalize', type='l')
            # l_min[idx_1_num] = lw[idx_1_num, 1]
            # l_max = torch.ones_like(num) * norm_or_restore(data_dir, torch.tensor([300.0], device=params['device']), mode='normalize', type='l')
            # lw[:, 0].clamp_(min=l_min, max=l_max)
            l_min = torch.ones_like(num) * norm_or_restore(data_dir, h_per_ar.clone(),  mode='normalize', type='l')
            w_unnorm = norm_or_restore(data_dir, lw[:, 1].clone().detach(), mode='restore', device=params['device'], type='w')
            l_min[idx_1_num] = norm_or_restore(data_dir, w_unnorm[idx_1_num].clone() + 10.0, mode='restore', device=params['device'], type='l')
            l_max = 1.0 * torch.ones_like(num)
            l_max[idx_1_num] = norm_or_restore(data_dir, torch.sqrt((p_unnorm - 25)**2 - w_unnorm[idx_1_num] ** 2), mode='normalize', type='l')
            l_max[idx_2_num] = norm_or_restore(data_dir, torch.sqrt((p_unnorm - 25)**2 - (2 * w_unnorm[idx_2_num] + 50.0) ** 2), device=params['device'], mode='normalize', type='l')
            l_max[idx_3_num] = norm_or_restore(data_dir, torch.sqrt((p_unnorm - 25)**2 - (3 * w_unnorm[idx_3_num] + 100.0) ** 2), device=params['device'],  mode='normalize', type='l')
            lw[:, 0].clamp_(min=l_min, max=l_max)
            
            # ######################################################################
            # # LW
            # h_unnorm = norm_or_restore(data_dir, h.clone().detach(), mode='restore', type='h')
            # h_per_ar = h_unnorm / params['ar_lim']
            # lw_unnorm = norm_or_restore(data_dir, lw.clone().detach(), mode='restore', type='lw', device=params['device'])
            # lw_unnorm.clamp_(min=max(h_per_ar, 40))
            
            # num = (torch.argmax(numP.clone().detach(), dim=-1, keepdim=True) + 1).squeeze()
            # gap = torch.zeros_like(num)
            # gap[num == 2] = 50; gap[num == 3] = 100
            
            # idx_1_num = torch.where(num == 1)[0]
            # idx_23_num = torch.where(torch.logical_or(num == 2, num == 3))[0]
            
            # w_unnorm_min = torch.ones_like(num) * h_per_ar
            # p_elong = p.tile(num.shape[0],)
            # w_unnorm_max = ((p_elong / math.sqrt(2) - gap) / num).squeeze()
            # lw_unnorm[:, 1].clamp_(min=w_unnorm_min, max=w_unnorm_max) 
            
            # ## L By innercircle & (P - 40 nm)
            # w = lw_unnorm[:, 1].clone()
            # l_unnorm_min = torch.zeros_like(w).squeeze()
            # l_unnorm_min[idx_1_num]    = w[idx_1_num]
            # l_unnorm_min[idx_23_num]   = h_per_ar
            
            # l_unnorm_max = torch.min(torch.sqrt(p_elong ** 2 - (num * w + gap) ** 2), (p_elong - 40))
            # lw_unnorm[:, 0].clamp_(min=l_unnorm_min, max=l_unnorm_max)
            # lw_norm = norm_or_restore(data_dir, lw_unnorm, mode='normalize', type='lw', device=params['device'])
            # lw.copy_(lw_norm)
            # ######################################################################
            
            # Rot
            rot_ = rot.clone().detach()
            rot_norm = rot_ / (torch.sum(rot_, dim=-1).unsqueeze(-1) + 1e-10)
            rot.data = rot_norm            
            
            return lw, numP, rot

