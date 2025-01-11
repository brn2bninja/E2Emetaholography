# Convolution and Fourier operations
import torch
import numpy as np

# Defines a metasurface, including phase and amplitude variation.
def define_metasurface(tm, tp, rot):
  epsilon = 1e-8
  tE0 = torch.sqrt(tm * 1.45 + epsilon)
  propagation_phase_term = tp * torch.pi + epsilon
  rotational_phase =  rot * torch.pi / 3
  geometric_phase_term = torch.tile(rotational_phase, (1, 1, 3))
  phase_term = (propagation_phase_term + geometric_phase_term)
  phase_term[phase_term > np.pi] -= 2 * np.pi
  return tE0 * torch.exp(1j * phase_term)

# Computes the hologram of a field
def compute_hologram(field):
  field_freq = torch.fft.fftshift(field, dim=(0, 1))
  out = torch.fft.ifft2(field_freq, dim=(0, 1))
  out = torch.fft.fftshift(out, dim=(0, 1))
  I = torch.abs(out) ** 2
  I_max = torch.max(torch.max(I, dim=0, keepdims=True)[0], dim=1, keepdims=True)[0]
  I_norm = I / I_max

  return I_norm

# # Computes the hologram of a field
# def compute_hologram(field):
#   # field_shifted = torch.fft.fftshift(field, dim=(0, 1))
#   field_freq = torch.fft.ifftshift(torch.fft.fft2(field, dim=(0, 1)), dim=(0, 1))
#   out = torch.fft.ifft2(field_freq, dim=(0, 1))
#   out = torch.fft.fftshift(out, dim=(0, 1))
#   I = torch.abs(out) ** 2
  
#   # Exclude the center pixels
#   num = I.shape[0] 
#   idx_xy = [i for i in range(num) if (abs(i-num//2) > 1)]
#   max_I = torch.max(I[idx_xy, :][:, idx_xy])  
#   I = I / max_I
#   return I