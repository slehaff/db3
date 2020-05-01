import numpy as np
import cv2
import argparse
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from PIL import Image
from jsoncloud import generate_json_pointcloud, generate_pointcloud


Pro = np.array([0,60,85]) # projector origin
ProI=np.array([0, 76.77369189, 22.39972965]) # Porjector origin projection
Cam = np.array([0,45, 20]) # camera origin
COr = np.array([0, 15, -55])
CamPro = np.array([15, 65])
nR = np.array([0,0,1])
Cmo = np.array([0, 45.11318, 19.5776])
cmo = np.array([0,  .11318, -.42239]) # camera image plane with cam = origo
nCam = np.array([0, 0.2588, 0.9659])
ProCam = np.array([0, 15, 65])
Cos11 = np.cos(11/360*2*np.pi)
Sin11 = np.sin(11/360*2*np.pi)
Cos15 = np.cos(15/360*2*np.pi)
Sin15 = np.sin(15/360*2*np.pi)
Pix = .001 # Pixel size
# PhiMin = 30 # Ref Min
# PhiMax = 255
RefLength = 21.782 # Ref Max
YrefStart = 48.846
deltaref = .2138

rwidth = 170
rheight = 170




def getcmi(x,y):
    cmi = np.array([0.,0.,0.])    
    cmi[0] = ((x-85)*Pix)+ cmo[0]
    cmi[1] = (((y-85)* Cos11*Pix)+ cmo[1])
    cmi[2] = (((y-85)* -Sin11*Pix)+ cmo[2])
    # cmi = np.add(Cam,cmi)
    return(cmi)

# def getcni(x,y):
#     cni = np.array([0.,0.,0.])    
#     cni[0] = Cmo[0]+(x-85)*Pix
#     cni[1] = Cmo[1]+ (y-85)* Cos15*Pix
#     cni[2] = Cmo[2]+ (y-85)* Sin15*Pix
#     return(cni)

# def makecmitable():
#     cmi = [0.,0.,0.]
#     cmitab = np.arange(170.0*170*3).reshape(170,170,3)
#     print('shape', cmitab.shape)
#     for x in range(170):
#         for y in range(170):cmi = np.array([0.,0.,0.])
#     return cmitab         


def getmref(cmi):
    mref = np.array([0,0,0])
    mref =  np.multiply(-55/np.dot(cmi,nR), cmi)
    mref = np.add(mref, Cam)
    return(mref)


def getProI():
    lengthc = abs(np.linalg.norm(Pro-Cmo)*np.cos(15/360*2*np.pi))
    c = [0, lengthc*np.sin(15/360*2*np.pi), -lengthc*np.cos(15/360*2*np.pi)]
    ProI = Pro + c
    print('ProI:', ProI, lengthc, c)
    return(ProI)


def getcni(cmi,phi, phimax, phimin):
    cni = np.array([0.,0.,0.])
    l = np.array([0.,0.,0.])
    Cmi = cmi+Cam
    l= (ProI- Cmi)/np.linalg.norm(ProI-Cmi)
    print('l:', l)
    deltaphi = (phimax-phimin)/170
    print('deltaphi:', deltaphi)
    cni[1] = Pix*phi/deltaphi*Cos15+cmo[1]*Cos15 #pixels
    print('cni[1]:', cni[1])
    t = (cni[1]-cmi[1])/l[1]
    print('Cmi:',Cmi)
    cni =  np.add(cmi ,t*l)
    print('t:',t,'cni:', cni)
    return(cni)



def getnref(cni):
    nref = np.array([0,0,0])
    nref = Cam + np.dot(nR, (Cam-COr))/np.dot(nR, (cni-Cam))*(cni-Cam)
    return(nref)

# cmi = np.array([0.,0.,0.])
#     y = rdash*RefLength + YrefStart
#     refLinePoint = [0, y, -35]
#     return(refLinePoint)


def getq(cmi):
    q= Cam + np.dot(ProCam, nR)/np.dot(cmi,nR)*cmi
    return(q)


def calcm3dpoint( mref, nref,cmi):
    distance = np.zeros((170,1))

    for i in range(0,100):
        normal = np.cross((Pro-[(i-50)*deltaref,nref[1],-35]),(mref-Cam))
        mag = np.sqrt(normal.dot(normal))
        normalized = normal/mag
        # print('norm', normalized)
        distance[i]= abs(np.dot(ProCam, normalized))
        if distance[i] ==0 :
            # print('found:', i, distance[i], distance[icmi = np.array([0.,0.,0.])-1])
            break
    # # print('min, max:', np.min(distance), np.max(distance))
    # print(distance)

    return()

def getm3dpoint(x,y,phi):
    cmi = getcmi(x,y)
    mref = getmref(cmi)
    nreflinepoint = getrefline(phi)
    # print(cmi, mref, nreflinepoint)
    m3dpoint = calcm3dpoint(mref, nreflinepoint, cmi)
    return(m3dpoint)


def threedpoints(unwrapfile):
    # unwrap = np.zeros((rheight, rwidth), dtype=np.float)
    # reference = np.zeros((rheight, rwidth), dtype=np.float)
    thdpoints =  np.zeros((rheight, rwidth), dtype=np.float) 
    unwrap = np.load(unwrapfile)
    # reference = np.load(referencefile)
    for j in range(rwidth):
        for i in range(rheight):
            print(getm3dpoint(i,j,unwrap[i,j]))

        print('j:', j)


def test3dpoints(unwrapfile, ref_unwrapfile):
    # Read 4 sample points:
    unwrap = np.load(unwrapfile)
    refunwrap = np.load(ref_unwrapfile)
    ProI = getProI()
    # print('50,50, unwrap(50,50):', unwrap[50,50])
    # print('70,50, unwrap(70,50):', unwrap[70,50])
    # print('50,70, unwrap(50,70):', unwrap[50,70])
    # print('100,100, unwrap(100,100):', unwrap[100,100])
    # print('150,150, unwrap(150,150):', unwrap[150,150])
    # print('150,100, unwrap(150,100):', unwrap[150,100])
    plist = [[0,0],[0,169],[169,0],[20,20],[50,50],[70,50],[50,70],[100,100],[100,150],[150,150],[150,100],[169,169]]
    cmilist =[]
    x= []
    y= []
    z= []
    x3= []
    y3= []
    z3= []
    Cma = [0,45,20]
    print(len(plist))
    for i in range(len(plist)):
        cmi = getcmi(plist[i][0], plist[i][1])
        # cmi = cmi + Cam
        print('cmi:', cmi)
        PhiMin = refunwrap[0,0]
        PhiMax = refunwrap[169,169]
        print('minmax:', PhiMax, PhiMin)
        getcni((cmi), unwrap[plist[i][0], plist[i][1]], PhiMin, PhiMax)
        # print(plist[i][0], plist[i][1], cmi, np.dot(cmi,nR))
        cmilist.append(cmi)
        x3.append( cmi[0])
        y3.append(cmi[1])
        z3.append(cmi[2])
        # x.append(0)
        # y.append(0)
        # z.append(0)
        mref = getmref(cmi)
        # print('mref:', mref)
        x.append( mref[0])
        y.append(mref[1])
        z.append(mref[2])
    # x.append( Cma[0])
    # y.append(Cma[1])
    # z.append(Cma[2])
    print(len(x))
    figure = plt.figure()
    ax = figure.add_subplot(111, projection = '3d')
    ax.scatter(x,y,z, c = 'r', marker = '.')
    ax.scatter(x3,y3,z3, c = 'b', marker = '.')
    x2 = [35,-35,0,0,0, ProI[0]]
    y2 = [0,0,0,45,60,ProI[1]]
    z2 = [0,0,0,20,85, ProI[2]]
    ax.scatter(x2,y2,z2, c = 'g', marker = 'x')
    ax.set_xlabel('Xaxis')
    ax.set_ylabel('Yaxis')
    ax.set_zlabel('Zaxis')
    plt.show()

# makecmitable()
unwfile = '/home/samir/Desktop/blender/pycode/scanplanes/render'+ str(1)+'/unwrap.npy'
ref_unwfile ='/home/samir/Desktop/blender/pycode/reference/scan_ref_folder/unwrap.npy'
test3dpoints(unwfile, ref_unwfile)