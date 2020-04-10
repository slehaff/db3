import numpy as np
import cv2
import argparse
import sys
import os
from PIL import Image
from jsoncloud import *


high_freq = 12
low_freq = 1
rwidth = 170
rheight = 170


focalLength = 938.0
centerX = 319.5
centerY = 239.5
scalingFactor = 5000

PI = np.pi


def unwrap_r(low_f_file, high_f_file, folder):
    filelow = folder + low_f_file
    filehigh = folder +  high_f_file
    wraplow = np.zeros((rheight, rwidth), dtype=np.float64)
    wraphigh = np.zeros((rheight, rwidth), dtype=np.float64)
    unwrapdata = np.zeros((rheight, rwidth), dtype=np.float64)
    im_unwrap = np.zeros((rheight, rwidth), dtype=np.float64)
    wraplow = np.load(filelow)  # To be continued
    wraphigh = np.load(filehigh)
    print('highrange=', np.ptp(wraphigh), np.max(wraphigh), np.min(wraphigh) )
    print('lowrange=', np.ptp(wraplow), np.max(wraplow), np.min(wraplow) )
    # print('high:', wraphigh)
    # print('low:', wraplow)
    
    unwrapdata = np.zeros((rheight, rwidth), dtype=np.float64)
    kdata = np.zeros((rheight, rwidth), dtype=np.int64)
    # wrap1data = cv2.GaussianBlur(wrap1data, (0, 0), 3, 3)
    # wrap2data = cv2.GaussianBlur(wrap2data, (0, 0), 4, 4)
    for i in range(rheight):
        for j in range(rwidth):
            kdata[i, j] = round((high_freq/low_freq * (wraplow[i, j])- wraphigh[i, j])/(2*PI))
            # unwrapdata[i,j] = wraphigh[i, j] + 2*PI*kdata[i, j]
    unwrapdata = np.add(wraphigh, np.multiply(2*PI,kdata) )
    print('kdata:', np.ptp(np.multiply(1,kdata)))
    print('unwrap:', np.ptp(unwrapdata))
    # print("I'm in unwrap_r")
    print('kdata:', kdata[::40, ::40])
    wr_save = folder + 'unwrap.npy'
    np.save(wr_save, unwrapdata, allow_pickle=False)
    # print(wr_save)
    # np.save('wrap24.pickle', wrap24data, allow_pickle=True)
    # unwrapdata = np.multiply(unwrapdata, 1.0)
    # unwrapdata = np.unwrap(np.transpose(unwrapdata))
    # unwrapdata = cv2.GaussianBlur(unwrapdata,(0,0),3,3)
    # unwrapdata = np.multiply(unwrapdata, 1.0)
    im_unwrap = np.divide(unwrapdata, 1)# np.max(unwrapdata)*255)
    # unwrapdata/np.max(unwrapdata)*255
    cv2.imwrite(folder + 'unwrap.png', im_unwrap)
    cv2.imwrite(folder + 'kdata.png', np.multiply(2*PI,kdata))

# def abs_unwrap_r(low_f, high_f, output_file,  folder):
#     filelow = folder + '/' + low_f
#     filehigh = folder + '/' + high_f
#     wraplow = np.load(filelow)  # To be continued
#     wraphigh = np.load(filehigh)
#     unwrapdata = np.zeros((rheight, rwidth), dtype=np.float)
#     kdata = np.zeros((rheight, rwidth), dtype=np.float)
#     # wrap1data = cv2.GaussianBlur(wrap1data, (0, 0), 3, 3)
#     # wrap2data = cv2.GaussianBlur(wrap2data, (0, 0), 4, 4)
#     for i in range(rheight):
#         for j in range(rwidth):
#             kdata[i, j] = round(
#                 (high_freq/low_f * wraplow[i, j] - 1.0 * wraphigh[i, j])/(2.0*np.pi))
#             unwrapdata[i, j] = 1.0 * wraplow[i, j] + 1.0*kdata[i, j]*np.pi

#     print(kdata[::20, ::20])
#     wr_save = folder + 'unwrap.npy'
#     np.save(wr_save, unwrapdata, allow_pickle=False)
#     print(wr_save)
#     # np.save('wrap24.pickle', wrap24data, allow_pickle=True)
#     unwrapdata = np.multiply(unwrapdata, 1.0)
#     # unwrapdata = np.unwrap(np.transpose(unwrapdata))
#     unwrapdata = cv2.GaussianBlur(unwrapdata,(0,0),3,3)
#     cv2.imwrite(folder + output_file, unwrapdata)


def deduct_ref(unwrap, reference, folder1, folder2):
    file1 = folder1 + '/' + unwrap
    file2 = folder2 + '/' + reference
    wrap_data = np.load(file1)  # To be continuedref_folder = '/home/samir/db3/scan/static/scan_folder/scan_ref_folder'

    ref_data = np.load(file2)
    net_data = np.subtract(ref_data, wrap_data)
    net_save = folder1 + 'abs_unwrap.npy'
    np.save(net_save, net_data, allow_pickle=False)
    print(net_save)
    # np.save('wrap24.pickle', wrap24data, allow_pickle=True)
    net_data = np.multiply(net_data, 1.0)
    cv2.imwrite(folder1 + 'abs_unwrap.png', net_data)


def unwrap(request):
    # folder = ScanFolder.objects.last().folderName
    folder = '/home/samir/db2/scan/static/scan_folder/scan_im_folder/'
    ref_folder = '/home/samir/db2/scan/static/scan_folder/scan_ref_folder'
    three_folder = '/home/samir/db2/3D/static/3scan_folder'
    unwrap_r('scan_wrap2.npy', 'scan_wrap1.npy', folder )
    deduct_ref('unwrap.npy', 'unwrap.npy', folder, ref_folder)
    # generate_color_pointcloud(folder + 'image1.png', folder + '/abs_unwrap.png', folder + '/pointcl.ply')
    generate_json_pointcloud(folder + 'image1.png', folder +
                             '/abs_unwrap.png', three_folder + '/pointcl.json')
    return render(request, 'scantemplate.html')


for i in range(5):

    folder = '/home/samir/Desktop/blender/pycode/scanplanes/render'+ str(i)+'/'
    ref_folder ='/home/samir/Desktop/blender/pycode/scans/scan_ref_folder' 
    unwrap_r('scan_wrap2.npy', 'scan_wrap1.npy', folder )
    deduct_ref('scan_wrap2.npy', 'scan_wrap2.npy', folder, ref_folder)
    # generate_json_pointcloud(folder + 'blenderimage2.png', folder + 'unwrap.png', folder +'pointcl.json')
    generate_pointcloud(folder + 'blendertexture.png', folder + '5mask.png' , folder + 'im_wrap1.png', folder +'pointcl-high.ply')
    generate_pointcloud(folder + 'blendertexture.png', folder + '5mask.png' , folder + 'im_wrap2.png', folder +'pointcl-low.ply')
    generate_pointcloud(folder + 'blendertexture.png', folder + '5mask.png', folder + 'unwrap.png', folder +'pointcl-unw.ply')
    generate_pointcloud(folder + 'blendertexture.png', folder + '5mask.png', folder + 'kdata.png', folder +'pointcl-k.ply')

