# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 14:10:47 2024

@author: cdrg
"""

import numpy as np
from okada85 import okada85
from coordinate import WGS84_2_TWD97
from scipy.optimize import nnls

class finite_fault_inversion:
    def __init__(self, sub_fualt_centroid, L_grid_nums, W_grid_nums):
        self.sub_fualt_centroid = sub_fualt_centroid
        self.sub_fault_nums = len(self.sub_fualt_centroid)
        self.L_grid_nums = L_grid_nums
        self.W_grid_nums = W_grid_nums
        
   #===================================================================
    def make_y(self, data_path, hor_filename, ver_filename):
        # y_vector : [E, N, U].T
        # sig_vector : [E_sig, N_sig, U_sig].T
        hor_data = finite_fault_inversion.import_data(data_path, hor_filename)[1:]
        ver_data = finite_fault_inversion.import_data(data_path, ver_filename)[1:]
        
        hor_data = np.array([a[0:-1] for a in hor_data], dtype=float)
        ver_data = np.array([a[0:-1] for a in ver_data], dtype=float)
        
        E = hor_data[:, 2]*1e-3
        N = hor_data[:, 3]*1e-3
        U = ver_data[:, 2]*1e-3
        
        lon = hor_data[:, 0]
        lat = hor_data[:, 1]
        
        lon_vector = np.tile(lon, 3)
        lat_vector = np.tile(lat, 3)
        
        E_sig = hor_data[:, 4]*1e-3
        N_sig = hor_data[:, 5]*1e-3
        U_sig = ver_data[:, 3]*1e-3
        
        y_vector = np.append(np.append(E, N), U)
        sig_vector = np.append(np.append(E_sig, N_sig), U_sig)
        
        Y = np.column_stack([lon_vector, lat_vector, y_vector, sig_vector])
        
        return Y
    
    #===================================================================
    def make_G(self, y_vector, sub_fault_geometry):        
        G = np.zeros([y_vector.shape[0], 3*self.sub_fault_nums])
        
        grid_point = {'x': y_vector[:int(y_vector.shape[0]/3),0], 'y': y_vector[:int(y_vector.shape[0]/3),1], \
                      'z': np.zeros(int(y_vector.shape[0]/3))}
        grid_TWD = WGS84_2_TWD97(grid_point['y'], grid_point['x'], grid_point['z']) 
        grid_point = {'x': grid_TWD[0], 'y': grid_TWD[1], 'z': grid_TWD[2]}
        
        for s in range(self.sub_fault_nums):
            ok = okada85(grid_point, self.sub_fualt_centroid[s], sub_fault_geometry)
            ux_green, uy_green, uz_green = ok.green_function()
             
            G[:,s] = np.concatenate([ux_green[0], uy_green[0], uz_green[0]])
            G[:,s + self.sub_fault_nums] = np.concatenate([ux_green[1], uy_green[1], uz_green[1]])
            G[:,s + 2*self.sub_fault_nums] = np.concatenate([ux_green[2], uy_green[2], uz_green[2]])
            '''
            G[:,s] = np.concatenate([ux_green[0], uy_green[0], uz_green[0]])
            G[:,s + self.sub_fault_nums] = np.concatenate([ux_green[1], uy_green[1], uz_green[1]])
            '''      
        return G
    
    #===================================================================
    # laplacian
    def make_L(self):      
        block_diag = np.diag(-4 * np.ones(self.L_grid_nums)) + \
                     np.diag(np.ones(self.L_grid_nums - 1), k=1) + \
                     np.diag(np.ones(self.L_grid_nums - 1), k=-1)
        
        L = np.kron(np.eye(self.W_grid_nums), block_diag) + \
            np.kron(np.diag(np.ones(self.L_grid_nums - 1), k=1), np.eye(self.W_grid_nums)) + \
            np.kron(np.diag(np.ones(self.L_grid_nums - 1), k=-1), np.eye(self.W_grid_nums))
        
        # bc
        for i in range(len(L)):
            L[i, i] = -L[i].tolist().count(1)
        
        L = np.kron(np.eye(3), L)
        
        return L
    
    #===================================================================
    # lstsq
    def fit(self, G, L, Y, alpha):
        sig = Y[:, -1]
        data = Y[:, -2]
        W = np.diag(1/sig)
        G_ori = G.copy()
        
        Gw = W @ G
        yw = W @ data
        y_pseudo = np.zeros(len(L))
        
        G = np.row_stack([Gw, alpha*L])
        y = np.concatenate([yw, y_pseudo])
        
        #mw = np.linalg.lstsq(G, y,rcond=None)[0]
        mw = nnls(G, y)[0]
        fit = G_ori @ mw
        
        #fit = fit[:len(yw)]
        return mw, fit
      
    #===================================================================    
    @staticmethod    
    def import_data(path, filename):
        data = []
        with open(path + filename) as file:
            for line in file:
                data.append(line.rstrip().split())
                
        return data

        