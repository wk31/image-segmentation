# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:01:13 2020

@author: w03798
"""


def rgb2xyz(x,y,z):
    x = x/255.0
    y = y/255.0
    z = z/255.0
    xnew = 0.4124564*x + 0.3575761*y + 0.1804375*z
    ynew = 0.2126729*x + 0.7151522*y + 0.0721750*z
    znew = 0.0193339*x + 0.1191920*y + 0.9503041*z
    return xnew, ynew, znew


def xyz2Lab(x,y,z):
    x = x/0.950456
    y = y/1.0
    z = z/1.088754
    
    if x>0.008856:
        xx = x**(1.0/3)
    else:
        xx = 7.787*x + 4.0/29
    
    if y>0.008856:
        yy = y**(1.0/3)
    else:
        yy = 7.787*y + 4.0/29
    
    if z>0.008856:
        zz = z**(1.0/3)
    else:
        zz = 7.787*z + 4.0/29
        
    L = 116.0*yy-16.0
    if L<0:L=0
    a = 500.0*(xx-yy)
    b = 200.0*(yy-zz)
    return L,a,b

def calcDeltaE(r1,g1,b1,r2,g2,b2):
    x1, y1, z1 = rgb2xyz(r1,g1,b1)
    x2, y2, z2 = rgb2xyz(r2,g2,b2)
    
    L1, A1, B1 = xyz2Lab(x1, y1, z1)
    L2, A2, B2 = xyz2Lab(x2, y2, z2)
    
    print(r1,g1,b1)
    print(r2,g2,b2)
    # print(L1,A1,B1)
    # print(L2,A2,B2)
    
    deltaE = ((L1-L2)**2 + (A1-A2)**2 + (B1-B2)**2)**0.5
    print(deltaE)

calcDeltaE(150,198,196,   153,199,199)















        