import numpy as np
from scipy.ndimage import map_coordinates

def lensdistort(I, k, **kwargs):
    # Check if input image is color or grayscale
    if len(I.shape) == 3:
        I2 = np.zeros_like(I)
        for i in range(3):
            I2[:,:,i] = imdistcorrect(I[:,:,i], k, **kwargs)
    elif len(I.shape) == 2:
        I2 = imdistcorrect(I, k, **kwargs)
    else:
        raise ValueError('Unknown image dimensions')
    
    return I2

def imdistcorrect(I, k, **kwargs):
    # Determine the size of the image to be distorted
    M, N = I.shape
    center = (N//2, M//2)
    
    # Create x-y points
    xi, yi = np.meshgrid(np.arange(N), np.arange(M))
    
    # Convert the mesh into a column vector of coordinates relative to the center
    xt = xi.flatten() - center[0]
    yt = yi.flatten() - center[1]
    
    # Convert the x-y coordinates to polar coordinates
    theta, r = np.arctan2(yt, xt), np.hypot(xt, yt)
    
    # Calculate the maximum vector (image center to image corner) to be used for normalization
    R = np.hypot(center[0], center[1])
    
    # Normalize the polar coordinate r to range between 0 and 1
    r = r / R
    
    # Apply the r-based transformation
    s = distortfun(r, k, kwargs.get('ftype', 4))
    
    # Un-normalize s
    s2 = s * R
    
    # Find a scaling parameter based on selected border type
    brcor = bordercorrect(r, s, k, center, R, kwargs.get('bordertype', 'crop'))
    
    s2 = s2 * brcor
    
    # Convert back to cartesian coordinates
    ut, vt = np.cos(theta) * s2, np.sin(theta) * s2
    
    u = ut.reshape(xi.shape) + center[0]
    v = vt.reshape(yi.shape) + center[1]
    
    # Perform image resampling
    I3 = map_coordinates(I, [v, u], order=3, mode=kwargs.get('interpolation', 'reflect'))
    
    return I3

def bordercorrect(r, s, k, center, R, bordertype):
    if k < 0:
        if bordertype == 'fit':
            x = r[0] / s[0]
        elif bordertype == 'crop':
            x = 1 / (1 + k * (min(center) / R) ** 2)
    elif k > 0:
        if bordertype == 'fit':
            x = 1 / (1 + k * (min(center) / R) ** 2)
        elif bordertype == 'crop':
            x = r[0] / s[0]
    return x

def distortfun(r, k, fcnum):
    if fcnum == 1:
        s = r * (1 / (1 + k * r))
    elif fcnum == 2:
        s = r * (1 / (1 + k * (r ** 2)))
    elif fcnum == 3:
        s = r * (1 + k * r)
    elif fcnum == 4:
        s = r * (1 + k * (r ** 2))
    return s
