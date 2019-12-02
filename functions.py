import random as r
import cv2
import numpy as np
import os
import csv

def cv(x1, n, d):
    cov = []
    for i in range(d):
        cov.append([sum([x1[k][i]*x1[k][j] for k in range(n)])/(n) for j in range(d)])
    return cov


def gen(n, d):
    x = [[r.uniform(0, 1) * n for i in range(d)] for j in range(n)]
    return x


def mean(x, n, d):
    return [sum(x[i][j] for i in range(n)) / n for j in range(d)]


def zeroMean(x, mn, n, d):
    return [[x[i][j]-mn[j] for j in range(d)] for i in range(n)]


def extract_bv(image):
    global fundus
    b,green_fundus,r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)

    # applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)

    # removing very small contours through area parameter noise removal
    ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    im2, contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
    	if cv2.contourArea(cnt) <= 200:
    		cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)

    # removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
    #vessels and also in an interval of area
    fundus_eroded = cv2.bitwise_not(newfin)
    xmask = np.ones(image.shape[:2], dtype="uint8") * 255
    x1, xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in xcontours:
    	shape = "unidentified"
    	peri = cv2.arcLength(cnt, True)
    	approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
    	if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
    		shape = "circle"
    	else:
    		shape = "veins"
    	if(shape=="circle"):
    		cv2.drawContours(xmask, [cnt], -1, 0, -1)

    finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)
    # blood_vessels = cv2.bitwise_not(finimage)
    blood_vessels = finimage
    return blood_vessels


def getMask(img):
    pathFolder = "./"
    filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]
    destinationFolder = "outputVessels"
    # if not os.path.exists(destinationFolder):
    # 	os.mkdir(destinationFolder)
    # for file_name in filesArray:
    # 	file_name_no_extension = os.path.splitext(file_name)[0]
    # fundus = cv2.imread("1.jpg",1)
    fundus=np.asarray(img)
    bloodvessel = extract_bv(fundus)
    cv2.imwrite("bloodvessel.png",bloodvessel)



# for i in range(len(gsImg)):
#     for pxl in range(len(gsImg[0])):
#         if gsImg[i][pxl] < 20:
#             gsImg[i][pxl] = 0
#         else:
#             gsImg[i][pxl] = 255
#
# plt.imshow(gsImg,cmap="gray")
# plt.show()
