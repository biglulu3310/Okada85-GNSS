# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:05:30 2024

@author: cdrg
"""

import numpy as np
from pyproj import Transformer

def cart_2_pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol_2_cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def WGS84_2_TWD97(lat, lon, alti):
    # TWD97 : {"Taiwan" : 3826, "Penghu, Kinmen & Matsu" : 3825}
    
    trans = Transformer.from_crs("epsg:4326", "epsg:3826")
    return trans.transform(lat, lon, alti)
    
def TWD97_2_WGS84(x, y, z):
    # TWD97 : {"Taiwan" : 3826, "Penghu, Kinmen & Matsu" : 3825}
    
    trans = Transformer.from_crs("epsg:3826", "epsg:4326")
    return trans.transform(x, y, z)