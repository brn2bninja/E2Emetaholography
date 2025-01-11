import numpy as np
import os
import math

# def gdstxtgen(fname, u_idx, r_idx, p, lwn, ph_step, ratio):
def gdstxtgen(fname, rot_ar, p, ratio, **vargs):
    
    if vargs['version'] == 'u_indicator':
        u_idx = vargs['u_idx']
        lwn = vargs['lwn']
        lw = lwn[:, [0, 1]]
        numP = lwn[:, 2]
        lw_2d = lw[u_idx, :]
        numP_2d = numP[u_idx]
    elif vargs['version'] == 'lw_array':
        lw_2d = vargs['lw']
        numP_2d = vargs['numP']
    
    num = lw_2d.shape[0]
    tmp_x = p * np.linspace(-num/2, num/2 - 1, num)
    tmp_y = tmp_x[::-1]
    
    if ratio == 1:
        ratio_str = str(int(ratio))
    else:
        ratio_str = '0' + str(int(ratio * 10))
        
    with open(fname + '_r=' + ratio_str + '.txt', 'w') as f:
        f.write('HEADER 3;\r\n')
        f.write('BGNLIB;\r\n')
        f.write(f'LIBNAME {fname}.txt;\r\n')
        f.write('UNITS 1.000000e+000 1.000000e-009;\r\n')
        f.write('BGNSTR;\r\n')
        f.write(f'STRNAME {fname};\r\n')
        for i in range(num):
            for j in range(num):
                x_o, y_o = tmp_x[i], tmp_y[j]
                theta = rot_ar[i, j] * (math.pi / 6)
                make_coordinates(x_o, y_o, theta, lw_2d[i, j], numP_2d[i, j], ratio, f)
        f.write('ENDSTR\r\n')
        f.write('ENDLIB\r\n')
    f.close()
    
    
def make_coordinates(x_o, y_o, theta, lw, n, ratio, f):
    l0 = lw[0]; w0 = lw[1]
    l = l0 * (l0>w0) + ratio * l0 * (l0<w0); w = w0 * (l0<w0) + ratio * w0 * (l0>w0) 
    
    def rotation(theta, x):
        result = (np.array([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ x.T).T
        return result
    
    def write_files(f, xy):
        f.write('BOUNDARY\r\n')
        f.write('LAYER 46;\r\n')
        f.write('DATATYPE 46;\r\n')
        f.write('XY\r\n')
        for idx in range(5):
            xy_idx = np.int64(np.round(xy[idx, :]))
            f.write(f"{xy_idx[0]}\t:\t{xy_idx[1]}\r\n")
        f.write('ENDEL\r\n')
    
    temp = make_coords(w, l)
    if n==1:
        temp = np.round(rotation(theta, temp))
        temp = temp + [x_o, y_o]
        write_files(f, temp)

    elif n==2:
        gap=50
        temp_xy1 = np.copy(temp)
        temp_xy1[:, 0] -= (gap+w)/2
        temp_xy1 = np.round(rotation(theta, temp_xy1))
        temp_xy1 = temp_xy1 + [x_o, y_o]
        write_files(f, temp_xy1)
        
        temp_xy2 = np.copy(temp)
        temp_xy2[:, 0] += (gap+w)/2
        temp_xy2 = np.round(rotation(theta, temp_xy2))
        temp_xy2 = temp_xy2 + [x_o, y_o]
        write_files(f, temp_xy2)

    elif n==3:
        gap=50
        temp_xy1 = np.copy(temp)
        temp_xy1[:, 0] -= (gap+w)
        temp_xy1 = np.round(rotation(theta,temp_xy1))
        temp_xy1 = temp_xy1 + [x_o, y_o]
        temp_xy2 = np.copy(temp)
        temp_xy2 = np.round(rotation(theta,temp_xy2))
        temp_xy2 = temp_xy2 + [x_o, y_o]
        temp_xy3 = np.copy(temp)
        temp_xy3[:, 0] += (gap+w)
        temp_xy3 = np.round(rotation(theta,temp_xy3))
        temp_xy3 = temp_xy3 + [x_o, y_o]
        write_files(f,temp_xy1)
        write_files(f,temp_xy2)
        write_files(f,temp_xy3)

    return None

def make_coords(w, l):
    temp = np.zeros([5, 2])
    temp[:, 0] = w/2 * np.array([1, -1, -1, 1, 1])
    temp[:, 1] = l/2 * np.array([1, 1, -1, -1, 1])

    return temp