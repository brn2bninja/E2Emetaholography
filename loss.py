import torch
   
def CT_loss(I, params):
    metric = torch.nn.MSELoss()
    target_img = params['target_images']
    color_weights = params['color_weights']
    [match_weight, unmatch_weight] = params['ct_weight']
    num_wl= I.shape[2]
    
    crit = 0.3
    match_loss = 0.0; unmatch_loss = 0.0
    for i in range(3):
        target_i = target_img[:,:,i]
        mask_i = torch.where(target_i > crit, 1, 0)
        rev_mask_i = torch.where(target_i > crit, 0, 1) 
        I_i = I[:,:,i]
        avg_target_ij = torch.mean(target_i)
        avg_I = torch.mean(I_i) 
        norm_I_ij = avg_target_ij * I_i / avg_I
        match_loss += match_weight * (color_weights[i] / sum(color_weights)) * metric((mask_i * norm_I_ij), target_i) / num_wl
        unmatch_loss += unmatch_weight * (color_weights[i] / sum(color_weights)) * torch.sum(rev_mask_i * norm_I_ij)
    return match_loss, unmatch_loss


# Per-Pixel loss
def Norm_loss(I, params):
    metric = torch.nn.MSELoss()
    normloss = 0.0
    I_tar = params['target_images']
    color_weights = params['color_weights']
    sum_batch_weight = sum(color_weights)
    for i in range(3):
        I_i = I[:,:,i]
        target_i = I_tar[:,:,i]
        avg_target_i = torch.mean(target_i)
        avg_I = torch.mean(I_i)
        norm_I_i = avg_target_i * I_i / avg_I
        normloss += params['norm_weight'] * (color_weights[i] / sum_batch_weight) * metric(norm_I_i, target_i)
    return normloss


# Perceptual loss (VGG19)
def P_loss(I, params):
    metric = torch.nn.MSELoss()
    I_tar = params['target_images']
    vgg_model = params['vgg_model']
    color_weights = params['color_weights']
    preprocessed_I  = torch.permute(I, (2, 0, 1)) * 255.0
    preprocessed_gt_img = torch.permute(I_tar, (2, 0, 1))*255.0

    G_layer_outs = [vgg_model[i](preprocessed_I) for i in range(len(vgg_model))]
    gt_layer_outs = [vgg_model[i](preprocessed_gt_img) for i in range(len(vgg_model))]
    loss = 0
    for i, weight in enumerate(color_weights):
        loss = loss + weight * sum([metric(G_layer_out[i] / 255., gt_layer_out[0] / 255. ) for G_layer_out, gt_layer_out in zip(G_layer_outs, gt_layer_outs)])
    return loss/3


# Sobel filter
def Edge_loss(I, params):
    metric = torch.nn.MSELoss()
    def sobel_filter(x):
        G_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        G_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        G_x = G_x.expand(1, 1, 3, 3)
        G_y = G_y.expand(1, 1, 3, 3)
        G_x = G_x.to(x.device)
        G_y = G_y.to(x.device)
        # dx = torch.nn.functional.conv2d(x, G_x, padding=1)
        # dy = torch.nn.functional.conv2d(x, G_y, padding=1)
        # return dx, dy
        dx = torch.nn.functional.conv2d(x.unsqueeze(0), G_x, padding=1).squeeze(0).squeeze(0)
        dy = torch.nn.functional.conv2d(x.unsqueeze(0), G_y, padding=1).squeeze(0).squeeze(0)
        return dx, dy
    
    edge_loss = 0.0
    target_img = params['target_images']
    for i, weight in enumerate(params['color_weights']):           
        dx, dy = sobel_filter(I[:, :, i:i+1].squeeze(-1))
        target_dx, target_dy = sobel_filter(target_img[:,:,i])
        edge_loss = edge_loss + weight * (torch.mean(metric(dx, target_dx)) + torch.mean(metric(dy, target_dy)))
    return edge_loss

# Spatial gradient loss
from pytorch_msssim import ssim
def SSIM_loss(I, params):
    target_img = params['target_images']
    s_loss = 1 - ssim(I.permute(0, 1, 2).unsqueeze(0), target_img.permute(0, 1, 2).unsqueeze(0))
    return params['ssim_weight'] * s_loss


def Eff_loss(tm_2d, params):
    eff_loss = 0.0
    color_weights = params['color_weights']
    sum_color_weight = sum(color_weights)
    for i in range(3):
        mean_tm_i = torch.mean(tm_2d[:,:,i])
        eff_loss += params['eff_weight'] * (color_weights[i] / sum_color_weight) * (1 - mean_tm_i)
    return eff_loss


def loss_functions(params):
    def hologram_loss_function(I, tm_2d):
        loss_components = {}
        # Compute losses
        if ('norm_weight' in params.keys()) and (params['norm_weight'] != 0):
            Norm_loss_value = Norm_loss(I, params)
            loss_components['norm'] = Norm_loss_value
        if ('ct_weight' in params.keys()) and (params['ct_weight'] != [0, 0]):
            CT_loss_value1, CT_loss_value2 = CT_loss(I, params)
            loss_components['ct1'] = CT_loss_value1; loss_components['ct2'] = CT_loss_value2
        if ('p_weight' in params.keys()) and (params['p_weight'] != 0):
            P_loss_value = P_loss(I, params)
            loss_components['p'] = P_loss_value
        if ('ssim_weight' in params.keys()) and (params['ssim_weight'] != 0):
            Spatial_loss_value = SSIM_loss(I, params)
            loss_components['ssim'] = Spatial_loss_value
        if ('eff_weight' in params.keys()) and (params['eff_weight'] != 0):
            Eff_loss_value = Eff_loss(tm_2d, params)
            loss_components['eff'] = Eff_loss_value
        if ('edge_weight' in params.keys()) and (params['edge_weight'] != 0):
            Edge_loss_value = Edge_loss(I, params)
            loss_components['edge'] = Edge_loss_value        
        total_loss = sum(loss_components.values())

        return total_loss, loss_components 
           
    return hologram_loss_function