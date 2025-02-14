# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:20:31 2024

@author: cdrg
"""

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from okada85 import okada85
from coordinate import WGS84_2_TWD97, TWD97_2_WGS84
import numpy as np
from finite_fault_inversion import finite_fault_inversion 

#===================================================================
#fault_centroid = {'x': 120.54, 'y': 22.92, 'z': -18e3}
fault_centroid = {'x': 120.5, 'y': 22.95, 'z': -18e3}
fault_TWD = WGS84_2_TWD97(fault_centroid['y'], fault_centroid['x'], fault_centroid['z']) 
fault_centroid = {'x': fault_TWD[0], 'y': fault_TWD[1], 'z': fault_TWD[2], 'depth': np.abs(fault_TWD[2])}

fault_geometry = {'strike':281, 'dip': 24, 'length': 45e3, 'width': 45e3}

L_grid_nums = 15
W_grid_nums = 15

# create finite rectangular fualt
sub_fault_x, sub_fault_y, sub_fault_z = okada85.create_fault(L_grid_nums, W_grid_nums, fault_centroid, fault_geometry)

sub_fault_x_flat = sub_fault_x.flatten()
sub_fault_y_flat = sub_fault_y.flatten()
sub_fault_z_flat = sub_fault_z.flatten()

okada85.plot_fault(sub_fault_x_flat, sub_fault_y_flat, sub_fault_z_flat)

#===================================================================
# sub_fault
sub_fault_centroid_list = [{'x': x, 'y': y, 'z': z, 'depth': abs(z)} for x, y, z in \
                           zip(sub_fault_x_flat, sub_fault_y_flat,sub_fault_z_flat)]

sub_fault_geometry = {'strike':281, 'dip': 24, \
                      'length': fault_geometry['length'] / L_grid_nums, \
                      'width': fault_geometry['width'] / W_grid_nums}
    
dislocation = {'slip': 0, 'rake': 0, 'opening': 0}

#===================================================================
# finite fault inversion
ffi = finite_fault_inversion(sub_fault_centroid_list, L_grid_nums, W_grid_nums)

offset_path = 'D:/RA_all/meinong/co_seismic_okada/data/'
hor_filename = 'hor_co_disp.txt'
ver_filename = 'ver_co_disp.txt'

Y = ffi.make_y(offset_path, hor_filename, ver_filename)
L = ffi.make_L()
G = ffi.make_G(Y, sub_fault_geometry)

alpha = 10
m, fit = ffi.fit(G, L, Y, alpha)

m_x = m[:int(len(m)/3)].reshape([L_grid_nums, W_grid_nums])
m_y = m[int(len(m)/3):2*int(len(m)/3)].reshape([L_grid_nums, W_grid_nums])
m_z = m[2*int(len(m)/3):].reshape([L_grid_nums, W_grid_nums])

fit_x = fit[:int(len(fit)/3)]
fit_y = fit[int(len(fit)/3):int(len(fit)/3)*2]
fit_z = fit[2*int(len(fit)/3):]

slip_vector = np.column_stack([m_x.flatten(), m_y.flatten()])

slip_vector_xy = np.array([np.array([(np.cos(-fault_geometry['strike_rad'] + np.deg2rad(450)), \
                             -np.sin(-fault_geometry['strike_rad']+np.deg2rad(450))),\
                            (np.sin(-fault_geometry['strike_rad']+np.deg2rad(450)), \
                             np.cos(-fault_geometry['strike_rad']+np.deg2rad(450)))]) @ a for a in slip_vector], dtype=float)

slip = np.sqrt(m_x**2+m_y**2)

#===================================================================
fig = plt.figure(figsize=(14,14))

ma = Basemap(projection='cyl',resolution="i",llcrnrlat=22.54, urcrnrlat=23.3, llcrnrlon=120, urcrnrlon=120.75)

ma.arcgisimage(server='http://server.arcgisonline.com/ArcGIS', \
               service='World_Shaded_Relief', xpixels=400, ypixels=None, dpi=200, verbose=False)

sub_fault_WGS = TWD97_2_WGS84(sub_fault_x, sub_fault_y, np.full(sub_fault_y.shape, 0)) 
contour = plt.contourf(sub_fault_WGS[1], sub_fault_WGS[0], slip, 20, cmap='rainbow', alpha=0.4)
cbar = plt.colorbar(contour, fraction=0.046, pad=0.05)
cbar.ax.tick_params(labelsize=20)

cbar.ax.set_aspect(20)
cbar.set_label('Slip(m)', fontsize=27, labelpad=30)

plt.plot(Y[:int(len(fit)/3),0], Y[:int(len(fit)/3),1], 'o', c='k', ms=8)

for i in range(int(len(fit)/3)):   
    plt.arrow(Y[i,0], Y[i,1], Y[i, -2]*2.5, Y[i+int(len(fit)/3),-2]*2.5, width=0.004, ec = 'k', fc='w')
    plt.arrow(Y[i,0], Y[i,1], fit_x[i]*2.5, fit_y[i]*2.5, width=0.004, ec = 'k', fc='r')

eq_centroid = {'x': 120.54, 'y': 22.92, 'z': -14.6e3}
plt.scatter(eq_centroid['x'], eq_centroid['y'], s=900, marker='*', facecolors='r', edgecolors='k')

plt.plot(sub_fault_WGS[1].flatten(), sub_fault_WGS[0].flatten(), 'o', c='k', ms=4)

plt.title("Dislocation", fontsize=45, pad=20)
plt.tight_layout()
plt.savefig("../output/dislocation.png", dpi=400)

#===================================================================
fig = plt.figure(figsize=(14,10))
ax = plt.subplot2grid((1, 1), (0, 0))

contour = ax.contourf(sub_fault_WGS[1], sub_fault_WGS[0], slip, 20, cmap='rainbow', alpha=0.5)
cbar = plt.colorbar(contour, ax=ax)

cbar.ax.tick_params(labelsize=20)

cbar.ax.set_aspect(20)
cbar.set_label('Slip(m)', fontsize=27, labelpad=30)

for i in range(len(slip_vector_xy)):  
    plt.arrow(sub_fault_WGS[1].flatten()[i], sub_fault_WGS[0].flatten()[i],\
              slip_vector_xy[i, 0]/1.5e1, slip_vector_xy[i, 1]/1.5e1, width=0.0025, ec='k', fc='w')

plt.scatter(eq_centroid['x'], eq_centroid['y'], s=900, marker='*', facecolors='r', edgecolors='k')

ax.tick_params(axis='both', which='major', direction='in',
                bottom=True, top=True, right=True, left=True,
                length=8, width=1.8, labelsize=18, pad=10)

ax.tick_params(axis='both', which='minor', direction='in',
                bottom=True, top=True, right=True, left=True,
                length=4, width=1.2, labelsize=18, pad=10)

[ax.spines[b].set_linewidth(1.5) for b in ['top', 'bottom', 'left', 'right']]
[ax.spines[b].set_color("black") for b in ['top', 'bottom', 'left', 'right']]

ax.set_title("Fault Slip", fontsize=40, pad=15)
plt.tight_layout()

plt.savefig("../output/slip.png", dpi=400)
