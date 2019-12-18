import cv2
import matplotlib.pyplot as plt
import numpy as np


folder = "/home/samir/db3/prototype/pylib/minicosines3/"
# Open the image
img1 = cv2.imread(folder +'0_cos.jpg')
img2 = cv2.imread(folder +'1_cos.jpg')
img3 = cv2.imread(folder +'2_cos.jpg')
img4 = cv2.imread(folder +'6_cos.jpg')
img5 = cv2.imread(folder +'7_cos.jpg')
img6 = cv2.imread(folder +'8_cos.jpg')
centertemplate = img1[500:520, 386:390]
cv2.imwrite(folder + 'centertemplate.png', centertemplate)
fringetemplate = img1[500:520, 375:379]
cv2.imwrite(folder + 'fringetemplate.png', fringetemplate)
onetemplate = img1[500:530, 187:203]
cv2.imwrite(folder + 'onetemplate.png', onetemplate)
twotemplate = img2[500:530, 187:203]
cv2.imwrite(folder + 'twotemplate.png', twotemplate)
threetemplate = img3[500:530, 187:203]
cv2.imwrite(folder + 'threetemplate.png', threetemplate)
fourtemplate = img4[500:530, 187:203]
cv2.imwrite(folder + 'fourtemplate.png', fourtemplate)
fivetemplate = img5[500:530, 187:203]
cv2.imwrite(folder + 'fivetemplate.png', fivetemplate)
sixtemplate = img6[500:530, 187:203]
cv2.imwrite(folder + 'sixtemplate.png', sixtemplate)


def match(img, template):
    h, w, _ = template.shape
    print(template.shape)
    print(w,h)
    res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        print(pt)
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
    return(img)
img = match(img4, fourtemplate)
# img = match(img, centertemplate)
# img= match(img, fringetemplate)
# cv2.imwrite(folder + 'res1.png',img1)
cv2.imwrite(folder + 'res4.png',img)