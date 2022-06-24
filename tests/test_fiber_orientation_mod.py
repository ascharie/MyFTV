# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 08:46:48 2022

@author: erica
"""

from myptv import fiber_orientation_mod
from numpy import array, linalg, transpose
import numpy as np
from myptv import imaging_mod
from math import pi

def test_fiber_orientation():
    '''

    Returns
    -------
    u : numpy array
        Unit vector denoting orientation of fiber.
    rms_error : float
        root mean square error of fiber_orientation result.

    '''
    
    
    c1 = imaging_mod.camera('1', (1000.,1000.))
    c2 = imaging_mod.camera('2', (1000.,1000.))
    c3 = imaging_mod.camera('2', (1000.,1000.))
    c4 = imaging_mod.camera('2', (1000.,1000.))
    
    c1.O = array([20., 20., 100.])
    c2.O = array([80., 20., 100.])
    c3.O = array([20., 40., 100.])
    c4.O = array([80., 0., 100.])


    c1.f = 100.
    c2.f = 100.
    c3.f = 100.
    c4.f = 100.

    c1.theta = array([0., 0, 0.])
    c2.theta = array([0., 0., 0.])
    c3.theta = array([0., 0., 0.])
    c4.theta = array([0., 0., 0.])
    c1.calc_R()
    c2.calc_R()
    c3.calc_R()
    c4.calc_R()

    imgsys = imaging_mod.img_system([c1,c2,c3,c4])

    a = array([20., 20., -60.])
    b = array([25., 18, -10.])
    
    delta = b-a
    u_exact = delta / np.linalg.norm(delta)

    aproj1 = c1.projection_ori(a)
    aproj2 = c2.projection_ori(a)
    aproj3 = c3.projection_ori(a)
    aproj4 = c4.projection_ori(a)
    
    A = array([aproj1,aproj2,aproj3,aproj4])
    
    bproj1 = c1.projection_ori(b)
    bproj2 = c2.projection_ori(b)
    bproj3 = c3.projection_ori(b)
    bproj4 = c4.projection_ori(b)

    B = array([bproj1,bproj2,bproj3,bproj4])
    
    
    fiber = fiber_orientation_mod.FiberOrientation(A, B)

    c,u,ori = fiber.image2fiber(imgsys.cameras)
    
    x_err = np.abs(u_exact[0]-u[0])
    y_err = np.abs(u_exact[1]-u[1])
    z_err = np.abs(u_exact[2]-u[2])
    
    errors = np.array([x_err,y_err,z_err])
    rms_error = np.sqrt((np.square(errors[0]) + np.square(errors[1]) + np.square(errors[2]))/3)
    
    ori = np.multiply(ori,180/pi)
    
    print(rms_error < 0.001)

    
    return u,rms_error
    















