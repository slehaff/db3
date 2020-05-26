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
ProI=np.array([0, 75.35837215, 27.68177481]) # Porjector origin projection
ProIT= [0., -31.31212532, 0.43730021]
Cam = np.array([0,45,20]) # camera origin
COr = np.array([0, 15, -55])
CamPro = np.array([15, 65])
nR = np.array([0,0,1])
nRT = np.array([0,0,-1])
Cmo = np.array([0, 45.11318, 19.5776])
cmo = np.array([0,  0, .4373]) # camera image plane with cam = origo
nCam = np.array([0, 0.2588, 0.9659])
ProCam = np.array([0, 15, 65])
Cos11 = np.cos(11/360*2*np.pi)
Sin11 = np.sin(11/360*2*np.pi)
Cos15 = np.cos(15/360*2*np.pi)
Sin15 = np.sin(15/360*2*np.pi)
Cos75 = np.cos(75/360*2*np.pi)
Sin75 = np.sin(75/360*2*np.pi)
Cos165 = np.cos(165/360*2*np.pi)
Sin165 = np.sin(165/360*2*np.pi)
Cos0 = np.cos(0/360*2*np.pi)
Sin0 = np.sin(0/360*2*np.pi)
Pix = .001 # Pixel size
PHINUL = 0.6778471585768825
# PhiMin = 30 # Ref Min
# PhiMax = 255
RefLength = 21.782 # Ref Max
YrefStart = 48.846
deltaref = .2138

rwidth = 170
rheight = 170



def pointransform(point):
    point = np.append(point,[1])
    R = np.array([[1,0,0,0], [0,Cos165,-Sin165,0],[0,Sin165,Cos165,0],[0,0,0,1]])
    T = np.array([[1,0,0,0], [0,1,0,-45],[0,0,1,-20],[0,0,0,1]])
    RT = np.matmul(R,T)
    transpoint = np.matmul(RT,point)
    transpoint = transpoint[0:3]
    return(transpoint)

def vectransform(vector):
    vector = np.append(vector,[0])
    R = np.array([[1,0,0,0], [0,Cos165,-Sin165,0],[0,Sin165,Cos165,0],[0,0,0,1]])
    T = np.array([[1,0,0,0], [0,1,0,-45],[0,0,1,-20],[0,0,0,1]])
    RT = np.matmul(R,T)
    transvect = np.matmul(RT,vector)
    transvect = transvect[0:3]
    return(transvect)



def getcmi(x,y):
    cmi = np.array([0.,0.,0.])    
    cmi[0] = ((x-85)*Pix)+ cmo[0]
    cmi[1] = (((y-85)*Pix)+ cmo[1])
    cmi[2] = (0 + cmo[2])
    return(cmi)


def getmref(cmi):
    mref = np.array([0,0,0])
    mref =  np.multiply(-55/np.dot(cmi,nRT), cmi)
    return(mref)


def getProI():
    ProT = pointransform(Pro)
    CmoT = pointransform(Cmo)
    vecA = CmoT-ProT
    vecB = np.array([0, 1, 0])
    ProI = CmoT + np.dot(vecA, vecB)/(np.linalg.norm(vecB))*vecB
    print('ProI:', ProI)
    return(ProI)


def getYref(phi, refunwrap):
    # refunwrap = refunwrap -PHINUL
    for i in range (170):
        delta = phi- np.mean(refunwrap[i,:])
        if delta < .0001  :
            break
        else:
            i+=1
    return(i)



def getcni(cmi,NI):
    cni = np.array([0.,0.,0.])
    # l = np.array([0.,0.,0.])
    # Cmi = cmi
    # l= (ProIT- Cmi)/np.linalg.norm(ProIT-Cmi)
    # slope = (Cmi[0]-ProIT[0])/(Cmi[1]-ProIT[1])
    cni[0] = (NI-85)*Pix
    # t = (cni[1]-cmi[1])/l[1]
    # cni = cmi +t*l
    cni[1] = cmi[1]# slope*(NI-85)*Pix +cmi[1] + slope*cmi[0]
    cni[2] = cmi[2]
    # print('cmi:', cmi,'cni:', cni)
    return(cni)


def getnref(cni):
    nref = np.array([0,0,0])
    nref =  np.multiply(-55/np.dot(cni,nRT), cni)
    return(nref)



def getq(cmi):
    ProT = pointransform(Pro)
    q=  np.dot(ProT, nRT)/np.dot(cmi,nRT)*cmi
    return(q)


def calcm3dpoint(que ,mref, nref,Cmi):
    ProT = pointransform(Pro)
    PQ = que-ProT
    QMref = mref-que
    MNref = nref-mref
    # nom = np.linalg.norm(PQ)*np.linalg.norm(QMref)
    # denom = (np.linalg.norm(PQ+MNref))
    # print('nom denom:', nom, denom)
    m3d=que + (np.linalg.norm(PQ)*np.linalg.norm(QMref))/(np.linalg.norm(PQ+MNref))*(Cmi)/np.linalg.norm(Cmi)
    m3d[2] = 4*m3d[2]

    # print('|PQ+MNref|:', np.linalg.norm(PQ+MNref),'|PQ|:', np.linalg.norm(PQ) ,'|m3d|:', np.linalg.norm(m3d), m3d)
    return(m3d)

# def getm3dpoint(x,y,phi):
#     cmi = getcmi(x,y)
#     mref = getmref(cmi)
#     nreflinepoint = getrefline(phi)
#     # print(cmi, mref, nreflinepoint)
#     m3dpoint = calcm3dpoint(mref, nreflinepoint, cmi)
#     return(m3dpoint)


# def threedpoints(unwrapfile):
#     # unwrap = np.zeros((rheight, rwidth), dtype=np.float)
#     # reference = np.zeros((rheight, rwidth), dtype=np.float)
#     thdpoints =  np.zeros((rheight, rwidth), dtype=np.float) 
#     unwrap = np.load(unwrapfile)
#     # reference = np.load(referencefile)
#     for j in range(rwidth):
#         for i in range(rheight):
#             print(getm3dpoint(i,j,unwrap[i,j]))

#         print('j:', j)

# def getxref(phi,reference):
#     for i in range(170):
#         while (phi -reference[i, 85])< .05:
#             i+=1

#     return(i)


def testtranspoints():
    L12 = np.array([0,10,20])
    L12 = pointransform(L12)
    print('L12:', L12)
    L12 = [0,20,10]
    L12 = pointransform(L12)
    print('L21:', L12)
    L12 = [0,-10,20]
    L12 = pointransform(L12)
    print('L-12:', L12)
    L12 = [0,-10,-20]
    L12 = pointransform(L12)
    print('L-1-2:', L12)
    L12 = [0,40,20]
    L12 = pointransform(L12)
    print('L42:', L12)
    L12 = [0,-20,40]
    L12 = pointransform(L12)
    print('L-24:', L12)


def testtransvects():
    V12 = np.array([0, Cos15, Sin15])
    V12T = vectransform(V12)
    print('V12T:', V12, V12T)
    V12 = np.array([0, 1, 0])
    V12T = vectransform(V12)
    print('V12T:', V12, V12T)
    V12 = np.array([0, 0, 1])
    V12T = vectransform(V12)
    print('V12T:', V12, V12T)


def make3dpoints(unwrapfile,unwrapfolder, ref_unwrapfile):
    points = np.zeros((rwidth, rheight), dtype=np.float)
    unwrap = np.load(unwrapfile)
    refunwrap = np.load(ref_unwrapfile)
    ProT = pointransform(Pro)
    print('ProT:', ProT)
    CmoT = pointransform(Cmo)
    print('CmoT:', CmoT)
    CamT = pointransform(Cam)
    print('CamT:', CamT)
    RefP = [0,0,-35]
    RefPT = pointransform(RefP)
    print('RefPT:', RefPT)
    ProIT = pointransform(ProI)
    print('ProIT:', ProIT)
    for i in range(rwidth):
        for j in range(rheight):
            cmi = getcmi(i, j)
            phi = unwrap[i,j]
            NI = getYref(phi,refunwrap )
            # print('NI:', NI, phi)
            cni = getcni(cmi,NI)
            que = getq(cmi)
            mref = getmref(cmi)
            nref = getnref(cni)
            points[i,j]= calcm3dpoint(que, mref, nref, cmi)[2]
        print(i,points[i,j])
    np.save(unwrapfolder + 'points.npy', points, allow_pickle=False)
    return(points)

def test3dpoints(unwrapfile, ref_unwrapfile):
    # Read 4 sample points:
    unwrap = np.load(unwrapfile)
    refunwrap = np.load(ref_unwrapfile)
    # refunwrap = refunwrap - refunwrap[85,85]
    ProT = pointransform(Pro)
    # print('ProT:', ProT)
    CmoT = pointransform(Cmo)
    # print('CmoT:', CmoT)
    CamT = pointransform(Cam)
    # print('CamT:', CamT)
    RefP = [0,0,-35]
    RefPT = pointransform(RefP)
    # print('RefPT:', RefPT)
    ProIT = pointransform(ProI)
    # print('ProIT:', ProIT)

    # print('150,100, unwrap(150,100):', unwrap[150,100])
    plist = [[0,85], [35,85],[85,85],[135,85], [169,85] ]#,[35,85],[45,85],[55,85],[65,85],[75,85],[85,85],[95,85],[105,85],[115,85],[125,85],[135,85],[145,85],[155,85]]
    # [85,5],[85,15],[85,25],[85,35],[85,45],[85,55],[85,65],[85,75],[85,85],[85,95],[85,105],[85,115],[85,125],[85,135],[85,145],[85,155]]
    cmilist =[]
    x= []
    y= []
    z= []
    x1= []
    y1= []
    z1= []
    mx3= []
    my3= []
    mz3= []
    nx4= []
    ny4= []
    nz4= []
    x5= [cmo[0]]
    y5= [cmo[1]]
    z5= [cmo[2]]
    # print(len(plist))
    for i in range(len(plist)):
        cmi = getcmi(plist[i][0], plist[i][1])
        phi = unwrap[plist[i][0], plist[i][1]]
        NI = getYref(phi,refunwrap )
        cni = getcni(cmi,NI)
        # cmi = cmi + Cam
        # print('cmi:', cmi)
        # PhiMin = refunwrap[0,85]
        # PhiMax = refunwrap[169,85]
        # xref= getxref(unwrap[plist[i][1], plist[i][0]], refunwrap)
        # print('phi:', unwrap[plist[i][1], plist[i][0]],  'xref:', xref)
        # print('minmax:', PhiMax, PhiMin)
        # cni = getcni(cmi, unwrap[plist[i][1], plist[i][0]], PhiMax, PhiMin)
        # print(plist[i][0], plist[i][1], cmi, np.dot(cmi,nR))
        cmilist.append(cmi)
        mx3.append( cmi[0])
        my3.append(cmi[1])
        mz3.append(cmi[2])
        nx4.append( cni[0])
        ny4.append(cni[1])
        nz4.append(cni[2])
        mref = getmref(cmi)
        x.append( mref[0])
        y.append(mref[1])
        z.append(mref[2])
        nref = getnref(cni)
        x1.append( nref[0])
        y1.append(nref[1])
        z1.append(nref[2])
        que= getq(cmi)
        x5.append( que[0])
        y5.append(que[1])
        z5.append(que[2])
        M3d= calcm3dpoint(que, mref, nref, cmi)
        x5.append( M3d[0])
        y5.append(M3d[1])
        z5.append(M3d[2])
        print('coords:',plist[i][1],',', plist[i][0],'NI:', NI, phi, 'cni:', cni,'cmi:', cmi,'mref:', mref,'nref:', nref,'que:', que, 'M3d:', M3d)
        # print('coords:',plist[i][1],',', plist[i][0],'NI:', NI, phi, 'M3d:', M3d)

    # l2= (cmilist[2]- cmilist[1])/np.linalg.norm(cmilist[2]- cmilist[1])
    # print('l2:', l2)
    # print(len(x))
    figure = plt.figure()
    ax = figure.add_subplot(111, projection = '3d')
    ax.scatter(x,y,z, c = 'r', marker = '.')
    ax.scatter(x1,y1,z1, c = 'b', marker = '.')
    ax.scatter(mx3,my3,mz3, c = 'r', marker = '.')
    ax.scatter(nx4,ny4,nz4, c = 'b', marker = '.')
    ax.scatter(x5,y5,z5, c = 'r', marker = '+')
    x2 = [0,0,0,0,ProT[0]]
    y2 = [-120,0,0,0,ProT[1]]
    z2 = [0,0,0,0,ProT[2]]
    # ax.scatter(x2,y2,z2, c = 'g', marker = 'x')
    ax.set_xlabel('Xaxis')
    ax.set_ylabel('Yaxis')
    ax.set_zlabel('Zaxis')
    plt.show()


def makereference(ref_file):
    reference = np.zeros((rwidth, rheight), dtype=np.float)
    for i in range(170):
        reference[i,:]= .9*i/170
    np.save(ref_unwfile, reference, allow_pickle=False)

def testfull(unwfile,folder):
    full = np.full((170,170),.5)
    np.save(unwfile,full, allow_pickle=False)
    cv2.imwrite(folder + 'unwrap.png', full*128)

# makecmitable()
for i in range(0,5):
    print('i:',i)
    unwfile = '/home/samir/Desktop/blender/pycode/scanplanes/render'+ str(i)+'/unwrap.npy'
    unwfolder = '/home/samir/Desktop/blender/pycode/scanplanes/render'+ str(i)+'/' 
    ref_unwfile ='/home/samir/Desktop/blender/pycode/reference/scan_ref_folder/unwrap.npy'
    test3dpoints(unwfile, ref_unwfile)
# testfull(unwfile, unwfolder)
# # make3dpoints(unwfile, ref_unwfile)
# makereference(ref_unwfile)
# testtranspoints()
# testtransvects()





####################################################################################################
# def makecmitable():
#     cmi = [0.,0.,0.]
#     cmitab = np.arange(170.0*170*3).reshape(170,170,3)
#     print('shape', cmitab.shape)
#     for x in range(170):
#         for y in range(170):cmi = np.array([0.,0.,0.])
#     return cmitab         

# def getcni(x,y):
#     cni = np.array([0.,0.,0.])    
#     cni[0] = Cmo[0]+(x-85)*Pix
#     cni[1] = Cmo[1]+ (y-85)* Cos15*Pix
#     cni[2] = Cmo[2]+ (y-85)* Sin15*Pix
#     return(cni)