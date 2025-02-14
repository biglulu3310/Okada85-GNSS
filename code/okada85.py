# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:54:05 2024

@author: cdrg
"""

import numpy as np
import matplotlib.pyplot as plt

from coordinate import WGS84_2_TWD97

class okada85(object):
    eps = 1e-14
    
    def __init__(self, grid_point, fault_centroid, fault_geometry, dislocation=None, nu=0.25):      
        self.grid_point = grid_point
        self.fault_centroid = fault_centroid    
        self.fault_geometry = fault_geometry   
        self.dislocation = dislocation
        self.nu = nu
        
        self._check_input_format()
        
        # degrees to radians
        self.fault_geometry['strike_rad'] = np.deg2rad(self.fault_geometry['strike'])
        self.fault_geometry['dip_rad'] = np.deg2rad(self.fault_geometry['dip'])
        if self.dislocation!=None:
            self.dislocation['rake_rad'] = np.deg2rad(self.dislocation['rake'])
    
    #========================================================================== 
    def _check_input_format(self):
        required_keys = {
            'grid_point': {'x', 'y', 'z'},
            'fault_centroid': {'x', 'y', 'z'},
            'fault_geometry': {'strike', 'dip', 'length', 'width'},
            'dislocation': {'slip', 'rake', 'opening'}
                        }
        if self.dislocation==None:
            required_keys = {
            'grid_point': {'x', 'y', 'z'},
            'fault_centroid': {'x', 'y', 'z'},
            'fault_geometry': {'strike', 'dip', 'length', 'width'}       
                        }
        
        for key, required in required_keys.items():
            obj = getattr(self, key)
            assert all(k in obj for k in required), f"Keys {required} are required in {key}."
       
    #==========================================================================
    # dislocation
    def green_function(self):    
        # make projection of fault centroid to x-y plane as origin
        self.grid_point = {key: (self.grid_point.get(key, 0) - self.fault_centroid.get(key, 0) \
                           if key in ['x', 'y'] else self.grid_point.get(key, 0))\
                           for key in set(self.grid_point) & set(self.fault_centroid)}
            
        # (X, Y, D) coordinate
        ec = self.grid_point['x'] + np.cos(self.fault_geometry['strike_rad']) * \
                                    np.cos(self.fault_geometry['dip_rad']) * self.fault_geometry['width'] / 2
                                    
        nc = self.grid_point['y'] - np.sin(self.fault_geometry['strike_rad']) * \
                                    np.cos(self.fault_geometry['dip_rad']) * self.fault_geometry['width'] / 2
        
        x = np.cos(self.fault_geometry['strike_rad']) * nc + \
            np.sin(self.fault_geometry['strike_rad']) * ec + \
            self.fault_geometry['length'] / 2
            
        y = np.sin(self.fault_geometry['strike_rad']) * nc - \
            np.cos(self.fault_geometry['strike_rad']) * ec + \
            np.cos(self.fault_geometry['dip_rad']) * self.fault_geometry['width']

        d = self.fault_centroid['depth'] + \
            np.sin(self.fault_geometry['dip_rad']) * self.fault_geometry['width'] / 2

        p = y * np.cos(self.fault_geometry['dip_rad']) + d * np.sin(self.fault_geometry['dip_rad'])
        q = y * np.sin(self.fault_geometry['dip_rad']) - d * np.cos(self.fault_geometry['dip_rad'])
        
        ux_green = [1/(2*np.pi)*(-self.chinnery(f, x, p, q) if f!=self.ux_tf \
                               else self.chinnery(f, x, p, q))\
                    for f in [self.ux_ss, self.ux_ds, self.ux_tf]]
    
        uy_green = [1/(2*np.pi)*(-self.chinnery(f, x, p, q) if f!=self.uy_tf \
                               else self.chinnery(f, x, p, q))\
                    for f in [self.uy_ss, self.uy_ds, self.uy_tf]] 
            
        uz_green = [1/(2*np.pi)*(-self.chinnery(f, x, p, q) if f!=self.uz_tf \
                               else self.chinnery(f, x, p, q))\
                    for f in [self.uz_ss, self.uz_ds, self.uz_tf]]
         
        ue_green = np.sin(self.fault_geometry['strike_rad']) * np.array(ux_green) - \
            np.cos(self.fault_geometry['strike_rad']) * np.array(uy_green)
            
        un_green = np.cos(self.fault_geometry['strike_rad']) * np.array(ux_green) + \
            np.sin(self.fault_geometry['strike_rad']) * np.array(uy_green)
            
        return ue_green, un_green, uz_green
    
    def forward(self, ux_green, uy_green, uz_green, dislocation):
        U1 = np.cos(self.dislocation['rake_rad']) * self.dislocation['slip']
        U2 = np.sin(self.dislocation['rake_rad']) * self.dislocation['slip']
        U3 = self.dislocation['opening']
        
        ux = U1 * ux_green[0] + U2 * ux_green[1] + U3 * ux_green[2]
        uy = U1 * uy_green[0] + U2 * uy_green[1] + U3 * uy_green[2]
        uz = U1 * uz_green[0] + U2 * uz_green[1] + U3 * uz_green[2]
    
        return ux, uy, uz
        
    #==========================================================================    
    # sub-func
    def chinnery(self, f, x, p, q):
        u = f(x, p, q) - \
            f(x, p - self.fault_geometry['width'], q) - \
            f(x - self.fault_geometry['length'], p, q) + \
            f(x - self.fault_geometry['length'], p - self.fault_geometry['width'], q)

        return u
   
    # strike-slip
    def ux_ss(self, xi, eta, q):
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = xi * q / (R * (R + eta)) + \
            self.I1(xi, eta, q, R) * np.sin(self.fault_geometry['dip_rad'])
            
        k = (q != 0)
        u[k] = u[k] + np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]) )
        
        return u
    
    def uy_ss(self, xi, eta, q):
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = (eta * np.cos(self.fault_geometry['dip_rad']) + 
             q * np.sin(self.fault_geometry['dip_rad'])) * q / (R * (R + eta)) + \
             q * np.cos(self.fault_geometry['dip_rad']) / (R + eta) + \
             self.I2(eta, q, R) * np.sin(self.fault_geometry['dip_rad'])
            
        return u

    def uz_ss(self, xi, eta, q):
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        db = eta * np.sin(self.fault_geometry['dip_rad']) - q * np.cos(self.fault_geometry['dip_rad'])
        u = (eta * np.sin(self.fault_geometry['dip_rad']) - \
             q * np.cos(self.fault_geometry['dip_rad'])) * q / (R * (R + eta)) + \
             q * np.sin(self.fault_geometry['dip_rad']) / (R + eta) + \
             self.I4(db, eta, q, R) * np.sin(self.fault_geometry['dip_rad'])
             
        return u
    
    #==========================================================================
    # dip-slip
    def ux_ds(self, xi, eta, q):
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = q / R - \
            self.I3(eta, q, R) * np.sin(self.fault_geometry['dip_rad']) * np.cos(self.fault_geometry['dip_rad'])
            
        return u
        
    def uy_ds(self, xi, eta, q):
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = ((eta * np.cos(self.fault_geometry['dip_rad']) + \
              q * np.sin(self.fault_geometry['dip_rad'])) * q / (R * (R + xi)) -
              self.I1(xi, eta, q, R) * np.sin(self.fault_geometry['dip_rad']) * np.cos(self.fault_geometry['dip_rad']))
            
        k = (q != 0)
        u[k] = u[k] + np.cos(self.fault_geometry['dip_rad']) * np.arctan((xi[k] * eta[k]) / (q[k] * R[k]))
        
        return u
       
    def uz_ds(self, xi, eta, q):
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        db = eta * np.sin(self.fault_geometry['dip_rad']) - q * np.cos(self.fault_geometry['dip_rad'])
        u = (db * q / (R * (R + xi)) -
             self.I5(xi, eta, q, R, db) * np.sin(self.fault_geometry['dip_rad']) * np.cos(self.fault_geometry['dip_rad']))
        
        k = (q != 0)
        u[k] = u[k] + np.sin(self.fault_geometry['dip_rad']) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]))
        
        return u 
    
    #==========================================================================
    # tensile-slip
    def ux_tf(self, xi, eta, q):
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = q ** 2 / (R * (R + eta)) - \
            self.I3(eta, q, R) * (np.sin(self.fault_geometry['dip_rad']) ** 2)
            
        return u
        
    def uy_tf(self, xi, eta, q):
        R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        u = - (eta * np.sin(self.fault_geometry['dip_rad']) - \
               q * np.cos(self.fault_geometry['dip_rad'])) * q / (R * (R + xi)) - \
               np.sin(self.fault_geometry['dip_rad']) * xi * q / (R * (R + eta)) - \
               self.I1(xi, eta, q, R) * (np.sin(self.fault_geometry['dip_rad']) ** 2)
            
        k = (q != 0)
        u[k] = u[k] + np.sin(self.fault_geometry['dip_rad']) * np.arctan((xi[k] * eta[k]) , (q[k] * R[k]))
        
        return u
       
    def uz_tf(self, xi, eta, q):
        R = np.sqrt(xi**2 + eta**2 + q**2)
        db = eta * np.sin(self.fault_geometry['dip_rad']) - q * np.cos(self.fault_geometry['dip_rad'])
        u = (eta * np.cos(self.fault_geometry['dip_rad']) + \
             q * np.sin(self.fault_geometry['dip_rad'])) * q / (R * (R + xi)) + \
             np.cos(self.fault_geometry['dip_rad']) * xi * q / (R * (R + eta)) - \
             self.I5(xi, eta, q, R, db) * np.sin(self.fault_geometry['dip_rad'])**2
             
        k = (q != 0) #not at depth=0?
        u[k] = u[k] - np.cos(self.fault_geometry['dip_rad']) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]))
        
        return u 
    
    #==========================================================================
    def I1(self, xi, eta, q, R):
        db = eta * np.sin(self.fault_geometry['dip_rad']) - q * np.cos(self.fault_geometry['dip_rad'])
        if np.cos(self.fault_geometry['dip_rad']) > self.eps:
            I = (1 - 2 * self.nu) * (- xi / (np.cos(self.fault_geometry['dip_rad']) * (R + db))) - \
                np.sin(self.fault_geometry['dip_rad']) / np.cos(self.fault_geometry['dip_rad']) * \
                self.I5(xi, eta, q, R, db)
        else:
            I = -(1 - 2 * self.nu) / 2 * xi * q / (R + db) ** 2
            
        return I
      
    def I2(self, eta, q, R):
        I = (1 - 2 * self.nu) * (-np.log(R + eta)) - \
            self.I3(eta, q, R)
            
        return I   
    
    def I3(self, eta, q, R):
        yb = eta * np.cos(self.fault_geometry['dip_rad']) + q * np.sin(self.fault_geometry['dip_rad'])
        db = eta * np.sin(self.fault_geometry['dip_rad']) - q * np.cos(self.fault_geometry['dip_rad'])
        
        if np.cos(self.fault_geometry['dip_rad']) > self.eps:
            I = (1 - 2 * self.nu) * (yb / (np.cos(self.fault_geometry['dip_rad']) * (R + db)) - np.log(R + eta)) + \
                np.sin(self.fault_geometry['dip_rad']) / np.cos(self.fault_geometry['dip_rad']) * \
                self.I4(db, eta, q, R)
                
        else:
            I = (1 - 2 * self.nu) / 2 * (eta / (R + db) + yb * q / (R + db) ** 2 - np.log(R + eta))
        return I
        
    def I4(self, db, eta, q, R):
        if np.cos(self.fault_geometry['dip_rad']) > self.eps:
            I = (1 - 2 * self.nu) * 1.0 / np.cos(self.fault_geometry['dip_rad']) * \
                (np.log(R + db) - np.sin(self.fault_geometry['dip_rad']) * np.log(R + eta))
        else:
            I = - (1 - 2 * self.nu) * q / (R + db)
            
        return I
    
       
    def I5(self, xi, eta, q, R, db):
        X = np.sqrt(xi**2 + q**2)
        if np.cos(self.fault_geometry['dip_rad']) > self.eps:
            I = (1 - 2 * self.nu) * 2 / np.cos(self.fault_geometry['dip_rad']) * \
                 np.arctan( (eta * (X + q*np.cos(self.fault_geometry['dip_rad'])) + \
                             X*(R + X) * np.sin(self.fault_geometry['dip_rad'])) /
                            (xi*(R + X) * np.cos(self.fault_geometry['dip_rad'])) )         
            I[xi == 0] = 0
        else:
            I = -(1 - 2 * self.nu) * xi * np.sin(self.fault_geometry['dip_rad']) / (R + db)
            
        return I 
    
    #==========================================================================
    # create finite rectangular fault
    # the fault centroid here refer to the whole fualt plain rather than single sub-fault(self.fault_centroid)
    @staticmethod
    def create_fault(L_grid_nums, W_grid_nums, fault_centroid, fault_geometry):
        fault_geometry['dip_rad'] = np.deg2rad(fault_geometry['dip'])
        fault_geometry['strike_rad'] = np.deg2rad(fault_geometry['strike'])
        
        z_top = fault_centroid['z'] + fault_geometry['width'] / 2 * np.sin(fault_geometry['dip_rad'])
        z_bot = fault_centroid['z'] - fault_geometry['width'] / 2 * np.sin(fault_geometry['dip_rad'])
        
        # assume strike is 0
        top_left = np.array([fault_centroid['x'] - \
                             fault_geometry['width'] / 2 * np.cos(fault_geometry['dip_rad']), \
                             fault_centroid['y'] + fault_geometry['length'] / 2])
            
        top_right = np.array([fault_centroid['x'] + \
                              fault_geometry['width'] / 2 * np.cos(fault_geometry['dip_rad']), \
                              fault_centroid['y'] + fault_geometry['width'] / 2])
            
        bot_left = np.array([fault_centroid['x'] - \
                             fault_geometry['width'] / 2 * np.cos(fault_geometry['dip_rad']), \
                             fault_centroid['y'] - fault_geometry['width'] / 2])
            
        bot_right = np.array([fault_centroid['x'] + \
                              fault_geometry['width'] / 2 * np.cos(fault_geometry['dip_rad']), \
                              fault_centroid['y'] - fault_geometry['width'] / 2])
        
        x_point = np.linspace(top_left[0], top_right[0], W_grid_nums+1)  
        y_point = np.linspace(top_left[1], bot_left[1], L_grid_nums+1) 

        # centroid                  
        grid_x, grid_y = np.meshgrid((x_point[:-1] + x_point[1:])/2, \
                                     (y_point[:-1] + y_point[1:])/2)

        X_flat = grid_x.flatten() - fault_centroid['x']
        Y_flat = grid_y.flatten() - fault_centroid['y']
        
        coordinates = np.vstack((X_flat, Y_flat))
        
        rotation_matrix = np.array([[np.cos(fault_geometry['strike_rad']), np.sin(fault_geometry['strike_rad'])], \
                                   [-np.sin(fault_geometry['strike_rad']), np.cos(fault_geometry['strike_rad'])]])
        
        rotated_coordinates = rotation_matrix @ coordinates
        
        X_rot = rotated_coordinates[0, :].reshape(grid_x.shape) + fault_centroid['x']
        Y_rot = rotated_coordinates[1, :].reshape(grid_y.shape) + fault_centroid['y']
          
        z_point = np.linspace(z_top, z_bot, W_grid_nums+1)
        z = np.tile((z_point[:-1] + z_point[1:])/2, (L_grid_nums, 1))
        return X_rot, Y_rot, z
    
    #==========================================================================  
    # plot fault 
    @staticmethod
    def plot_fault(x, y, z):
        fig0 = plt.figure(figsize=(15, 15))
        ax0 = plt.subplot2grid((1, 1),(0, 0), projection='3d')
        
        #ax0.plot(x, y, z, marker='o', color='k', ms=10) 
        ax0.scatter(x, y, z, marker='o', color='k', s=50) 
        ax0.set_xlabel('x', fontsize=25, labelpad=40)
        ax0.set_ylabel('y', fontsize=25, labelpad=40)
        ax0.set_zlabel('z', fontsize=25, labelpad=40)
        [ax0.spines[a].set_linewidth(3.5) for a in ['top','bottom','left','right']]
        [ax0.spines[a].set_color("black") for a in ['top','bottom','left','right']]
    
        ax0.tick_params(axis='both', which='major',direction='in',\
                            bottom=True,top=True,right=True,left=True,\
                            length=21, width=4.5, labelsize=20, pad=15)
        
        ax0.tick_params(axis='both', which='minor',direction='in',\
                    bottom=True,top=True,right=True,left=True,\
                    length=9, width=3, labelsize=20, pad=15)
        
        ax0.view_init(20, 300)
        plt.tight_layout()  

#==============================================================================



    
    
    
    