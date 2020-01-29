from __future__ import unicode_literals
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .forms import CalibrateForm
from .pylib.servers import messenger
from .pylib.servers.picam import new_receiver_thread
import os
from calibrate.pylib.makewrap import *
from calibrate.pylib.unwrap import *
# from scan.pylib.pointcloud import *
from calibrate.pylib.jsoncloud import *
from calibrate.pylib.calculate import *
from calibrate.pylib.jsoncloud import *

import shutil
import json
from distutils.dir_util import copy_tree
import os
import shutil

# def calibrate(request):
#     return render(request, 'calibrate.html' )

def scan_wrap(folder):
    print('scan_wrap started')
    print(folder)
    take_wrap(folder, 'scan_wrap1.npy', 'im_wrap1.png', 'image', 2)
    print('low done!')
    take_wrap(folder, 'scan_wrap2.npy', 'im_wrap2.png', 'image', 5)


# def unwrap(request):
#     # folder = ScanFolder.objects.last().folderName
#     folder = '/home/samir/db2/scan/static/scan_folder/scan_im_folder/'
#     ref_folder = '/home/samir/db2/scan/static/scan_folder/scan_ref_folder'
#     three_folder = '/home/samir/db2/3D/static/3scan_folder'
#     unwrap_r('scan_wrap2.npy', 'scan_wrap1.npy', folder )
#     deduct_ref('unwrap.npy', 'unwrap.npy', folder, ref_folder)
#     # generate_color_pointcloud(folder + 'image1.png', folder + '/abs_unwrap.png', folder + '/pointcl.ply')
#     generate_json_pointcloud(folder + 'image1.png', folder +
#                              '/abs_unwrap.png', three_folder + '/pointcl.json')
#     return render(request, 'calibrate.html')

def unwrap2(foldernumber):
    folder = '/home/samir/db3/calibrate/static/calibrate_folder/calscans/cal_im_folder'+ str(foldernumber)+'/'
    unwrap_r('scan_wrap2.npy', 'scan_wrap1.npy', folder )
    generate_pointcloud(folder + 'image1.png', folder + 'unwrap.png', folder +'pointcl.ply')

    return

# def calcalc():
#     return

def calibrate(request):
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        sform = CalibrateForm(request.POST)
        # check whether it's valid:
        if 'CalCalc' in request.POST:
            print('CalCalc pressed')
            folder = '/home/samir/db3/calibrate/static/calibrate_folder/calscans/'
            calculate(folder)
            # worldtargets(folder)
        else:
            
            if sform.is_valid():
                scandata = sform.cleaned_data['calibration_type']
                print(scandata)
                if scandata == 99:
                    messenger.gamma_mess()
                    print('take gamma')
                    take_gamma()
                else:
                    folder = '/home/samir/db3/calibrate/static/calibrate_folder/calscans/cal_im_folder'+ str(scandata)+'/'             
                    # _ = new_receiver_thread('1', folder=folder)
                    # print('scan receiver started')
                    # print('start take')
                    try:
                        shutil.rmtree(folder)
                    except OSError as error:
                        print(error)
                    os.mkdir(folder)
                    messenger.scan_mess()
                    take_scan(scandata)
                    unwrap2(scandata)
                    print('calibrate scan')

                    # process the data in form.cleaned_data as required
                    # ...
                    # redirect to a new URL:
                    sform = CalibrateForm()
                    # return HttpResponse('/thanks/')


 
    else:
        'empty form'
        sform = CalibrateForm()
    context = {"calibrate_page": "active", 'sform': sform}
    return render(request, 'calibrate.html', context)


def take_scan(foldernumber):
    print('calibrate scan')
    folder = '/home/samir/db3/calibrate/static/calibrate_folder/calscans/cal_im_folder'+ str(foldernumber)+'/'
    t = new_receiver_thread('1', folder=folder)
    print('scan receiver started')
    print('start take')
    # scan_mess()
    t.join()
    scan_wrap(folder=folder)
    return 

def take_gamma():
    print('take gamma!')
    folder = '/home/samir/db3/scan/static/scan_folder/gamma_folder/'
    t = new_receiver_thread('1', folder=folder)
    print('gamma receiver started')
    print('gamma take')
    # scan_mess()
    t.join()
    # scan_wrap(folder=folder)

    return

# def gamma_cal(request):
#     print('3D')
#     folder = '/home/samir/db2/scan/static/scan_folder/gamma_im_folder/'
#     t = new_receiver_thread('1', folder=folder)
#     gamma_mess()
#     t.join()
#     # gam = gamma_curve(folder)
#     compensate_gamma(folder + 'image1.png')
#     return render(request, 'scantemplate.html')
