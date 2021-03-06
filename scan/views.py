from __future__ import unicode_literals
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .forms import ScanForm
from .pylib.servers import messenger
from .pylib.servers.picam import new_receiver_thread
import os
from scan.pylib.makewrap import *
from scan.pylib.scanunwrap import *
# from scan.pylib.unwrap import *
# from scan.pylib.pointcloud import *
from scan.pylib.jsoncloud import *

import shutil
import json
from distutils.dir_util import copy_tree
import os

def scan_wrap(folder):
    print('scan_wrap started')
    print(folder)
    take_wrap(folder, 'scan_wrap1.npy', 'im_wrap1.png', 'image', 2)
    take_wrap(folder, 'scan_wrap2.npy', 'im_wrap2.png', 'image', 5)
    take_wrap(folder, 'scan_wrap3.npy', 'im_wrap3.png', 'image', 8)
    take_wrap(folder, 'scan_wrap4.npy', 'im_wrap4.png', 'image', 11)
    avewrap(folder,'scan_wrap1.npy','scan_wrap2.npy','hf_wrap.npy')
    avewrap(folder,'scan_wrap3.npy','scan_wrap4.npy','lf_wrap.npy')


def unwrap(request):
    # folder = ScanFolder.objects.last().folderName
    folder = '/home/samir/db2/scan/static/scan_folder/scan_im_folder/'
    ref_folder = '/home/samir/db2/scan/static/scan_folder/scan_ref_folder'
    three_folder = '/home/samir/db2/3D/static/3scan_folder'
    unwrap_r('scan_wrap2.npy', 'scan_wrap1.npy', folder )
    deduct_ref('unwrap.npy', 'scan_wrap2.npy', folder, ref_folder)
    # generate_color_pointcloud(folder + 'image1.png', folder + '/abs_unwrap.png', folder + '/pointcl.ply')
    generate_json_pointcloud(folder + 'image1.png', folder +
                             '/abs_unwrap.png', three_folder + '/pointcl.json')
    return render(request, 'scantemplate.html')


def unwrap2():
    folder = '/home/samir/db3/scan/static/scan_folder/scan_im_folder/'
    ref_folder = '/home/samir/db3/scan/static/scan_folder/scan_ref_folder'
    unwrap_r('lf_wrap.npy', 'hf_wrap.npy', folder )
    deduct_ref('unwrap.npy', 'unwrap.npy', folder, ref_folder)
    generate_json_pointcloud(folder + 'image2.png', folder + 'unwrap.png', folder +'pointcl.json')
    generate_pointcloud(folder + 'image2.png', folder + 'unwrap.png', folder +'pointcl.ply')
    
    return

def get_http(request):
    #Process incoming http live get
    if request.method == 'GET':
        print('got get!')
    return('It Worked!!!')

def scan(request):
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        sform = ScanForm(request.POST)
        # check whether it's valid:
        if sform.is_valid():
            scandata = sform.cleaned_data['scan_type']
            print(scandata)
            if scandata == 4:
                messenger.gamma_mess()
                print('take gamma')
                take_gamma()
            else:
                messenger.scan_mess()
                take_scan()
                # unwrap2()
                # process the data in form.cleaned_data as required
                # ...
                # redirect to a new URL:
                sform = ScanForm()
                # return HttpResponse('/thanks/')
    else:
        'empty form'
        sform = ScanForm()
    context = {"scan_page": "active", 'sform': sform}
    return render(request, 'scan.html', context)


def take_scan():
    print('take scan')
    folder = '/home/samir/db3/scan/static/scan_folder/scan_im_folder/'
    n = len(os.listdir(folder))
    print('n:',n)
    folder = folder+'render'+ str(n)+'/'
    os.mkdir(folder)
    t = new_receiver_thread('1', folder=folder)
    print('scan receiver started')
    print('start take')
    # scan_mess()
    t.join()
    # scan_wrap(folder=folder)
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


