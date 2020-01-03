
import socket
import cv2
import numpy as np
import struct
import threading
import shutil
import os


def mov_files(dir_to):
    path = '/home/samir/db3/scan/static/scan_folder/im_folder/'
    moveto = dir_to
    files = os.listdir(path)
    print(files)
    files.sort()
    for f in files:
        src = path + f
        dst = moveto + f
        shutil.move(src, dst)


def rcv_all(sock, count):
    buf = b''
    while count:
        new_buf = sock.recv(count)
        if not new_buf:
            return None
        buf += new_buf
        count -= len(new_buf)
    return buf




def new_receive_pi_data(n, to_folder):
    print(to_folder)
    print('new receive called')
    server_socket = socket.socket()
    server_socket.bind(('0.0.0.0', 8001))
    server_socket.listen(0)
    conn, adr = server_socket.accept()
    i = 0
    try:
        while True:
            image_len = struct.unpack(
                '<L', conn.recv(struct.calcsize('<L')))[0]
            if not image_len:
                break
            string_data = rcv_all(conn, int(image_len))
            data = np.fromstring(string_data, dtype='uint8')
            dec_img = cv2.imdecode(data, 1)
            # dec_img = cv2.flip(dec_img, 1)  # Invert image
            dec_img2 = cv2.flip(dec_img, 1)  # Invert image
            # dec_img = cv2.flip(dec_img, 0)  # flip horisontal
            # dec_img = cv2.flip(dec_img, 0)  # Invert image
            print('received image!')
            #dec_img = dec_img[100:400, 30:530]
            i += 1
            folder = '/home/samir/db3/scan/static/scan_folder/im_folder/'
            cv2.imwrite(folder + '/image' + str(i-1)+'.png', dec_img2)
            print('Calibrate i=', i, image_len, dec_img.shape)
    finally:
        conn.close()
        server_socket.close()
        mov_files(to_folder)
        print('closed')



def new_receiver_thread(n, folder):
    t = threading.Thread(target=new_receive_pi_data,
                         args=(n, folder),
                         name='T1')
    t.start()
    return t




# *******************************Obsolete Files !!! ********************************************





def make_receiver_thread(n):
    t = threading.Thread(target=receive_pi_data,
                         args=n,
                         name='T1')
    t.start()



def receive_pi_data(n):
    server_socket = socket.socket()
    server_socket.bind(('0.0.0.0', 8001))
    server_socket.listen(0)
    conn, adr = server_socket.accept()
    i = 0
    try:
        while True:
            image_len = struct.unpack(
                '<L', conn.recv(struct.calcsize('<L')))[0]
            if not image_len:
                break
            string_data = rcv_all(conn, int(image_len))
            data = np.fromstring(string_data, dtype='uint8')
            dec_img = cv2.imdecode(data, 1)
            dec_img = cv2.flip(dec_img, 1)  # Invert image
            dec_img = cv2.flip(dec_img, 1)  # Invert image
            dec_img = cv2.flip(dec_img, 0)  # flip horisontal
            # dec_img = dec_img[25:265, 180:630]
            i += 1
            folder = '/home/samir/db3/prototype/static/scan_folder/im_folder/'
            cv2.imwrite(folder + '/image' + n + str(i)+'.png', dec_img)
            print(image_len)
    finally:
        conn.close()
        server_socket.close()
        print('closed')



# folder = '/home/samir/danbotsIII/scan/static/scan/scanfolders/folder' + str(32)
# receive_pi_data()
