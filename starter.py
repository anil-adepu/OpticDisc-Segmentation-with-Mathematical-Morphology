import functions as f
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import scipy.ndimage as sc

def rgbToGsImg(zm,pc,r,c):
    out = np.matmul(zm,pc)
    mini,mx=min(out),max(out)
    print(mini,mx)
    ch=" ,  "
    for i in range(len(out)):
        out[i] = out[i]-mini

    mini,mx=min(out),max(out)
    print(mini,mx)

    img = []
    for i in range(r):
        row =[]
        for j in range(c):
            row.append(out[i*c+j])
        img.append(row)

    for i in range(r):
        for j in range(c):
            img[i][j] = np.uint8(round( ((img[i][j])/mx)*255))
    return img

def enhanceImg(img,r,c):
    uMin,uMax = {0,255}
    e=2
    w = int(round(0.15*(max(r,c))))

    print(w)
    if w%2==0:
        w+=1
    inc = int(w/2)
    R,C = { r + 2*inc, c + 2*inc }
    paddedImg = [[0 for i in range(C)]for j in range(R)]

    fmin,fmax = {100000000,-100000000}
    for i in range(r):
        for j in range(c):
            if img[i][j] > fmax:
                fmax = img[i][j]
            if img[i][j] < fmin:
                fmin = img[i][j]
            paddedImg[i+inc][j+inc] = img[i][j]
    print(fmin,fmax)
    for i in range(inc, r+inc):
        for j in range(inc, c+inc):
            sum=0
            for i1 in range(i-inc,i+inc+1):
                for j1 in range(j-inc,j+inc+1):
                    sum += paddedImg[i1][j1]
            mean = float(sum)/float(w*w)
            if(paddedImg[i][j] <= mean):
                # print(img[i-inc][j-inc])
                img[i-inc][j-inc] = uMin + ((uMax)*((paddedImg[i][j]-fmin)**e))/(2*((mean-fmin)**e))
                # print("...hiii  -",end="")
                # print(img[i-inc][j-inc])
            else:
                img[i-inc][j-inc] = uMax - ((uMax)*((paddedImg[i][j]-fmax)**e))/(2*((mean-fmax)**e))
    return img


def inPaintImg(img,r,c):

    return 1


eyeImg = plt.imread("g0023.jpg")
eyeImg = np.array(eyeImg)
rowLen = len(eyeImg)
colLen = len(eyeImg[0])
print(rowLen)
print(colLen)

rows = []
for row in eyeImg:
    for pixel in row:
        pxl = []
        for rgb in pixel:
            pxl.append(rgb)
        rows.append(pxl)
# print(rows)
# cv2.imshow("img",temp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.imshow(eyeImg)
plt.title("original image")
plt.show()

mn = f.mean(rows, len(rows), 3)

zm = f.zeroMean(rows, mn, len(rows), 3)
cov = f.cv(zm, len(rows), 3)
eigMat = np.linalg.eig(cov)
print(eigMat, end='\n\n\n')
eigVal = [[eigMat[0][i], eigMat[1][i]] for i in range(3)]
# eigVal.sort(key=lambda x: x[0], reverse=True)
# print((eigVal))

eigSum = sum(eigMat[0][i] for i in range(3))
#print(eigSum)

k = 1
# sum = 0
# for i in eigMat[0]:
#     sum += i
#     k+=1
#     if float(sum) / float(eigSum) >=0.9:
#         break
# print(k)

gsAll=[]

for i in range(k):
    gsImg = rgbToGsImg(zm,eigMat[1][i],rowLen,colLen)


    #print(gsImg)

    plt.imshow(gsImg,cmap="gray")
    plt.title("PCA output")
    plt.show()

    # enhance

    # # gsImgEn = enhanceImg(gsImg,rowLen,colLen)
    # plt.imshow(gsImgEn,cmap="gray")
    # plt.show()

    f.getMask(eyeImg)
    mask = cv2.imread("bloodvessel.png",0)
    mask = sc.binary_dilation(mask)
    mask = sc.binary_dilation(mask)
    mask = sc.binary_dilation(mask)

    # print(mask[0])
    msk = []
    for i in range(len(mask)):
        a = []
        for j in range(len(mask[0])):
            if mask[i][j]==False:
                a.append((np.uint8(0)))
            else:
                a.append(np.uint8(1))#mask[i][j]=np.uint8(1)
        msk.append(a)
    plt.imshow(msk,cmap="gray")
    plt.title("mask")
    plt.show()

    # cv2.imshow("img",mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # # inpaint

    for i in range(5):
        gsImgEnIn = cv2.inpaint(np.asarray(gsImg),np.asarray(msk),30,cv2.INPAINT_TELEA)
        gsImg = gsImgEnIn
    # gsImgEnIn = inPaintImg(gsImgEn,rowLen,colLen)

    # plt.imshow(gsImg,cmap="gray")
    # plt.show()

    gsAll.append(gsImg)

print(len(gsAll))
for i in range(len(gsAll[0])):
    for j in range(len(gsAll[0][0])):
        gsAll[0][i][j] = max(gsAll[t][i][j] for t in range(k))
plt.imshow(gsAll[0],cmap="gray")
plt.title("final inpainted image")
plt.show()

gsImgEnIn=gsAll[0]

# kernel = np.ones((3,3), np.uint8)
# img_erosion = cv2.erode(gsImgEnIn, kernel, iterations=5)
# img_dilation = cv2.dilate(gsImgEnIn, kernel, iterations=5)

# # print(img_erosion[0])

# plt.imshow(img_dilation,cmap="gray")
# plt.title("dilation of inpainted image")
# plt.show()
# plt.imshow(img_erosion,cmap="gray")
# plt.title("erosion of inpainted image")
# plt.show()

# # img_gradient = img_dilation - img_erosion
# img_gradient = cv2.subtract(img_dilation,img_erosion)
# plt.imshow(img_gradient,cmap="gray")
# plt.title("gradient(dilation-erosion) of inpainted image")
# plt.show()


img = eyeImg
gray = gsImgEnIn
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
plt.imshow(thresh, cmap="gray")
plt.title("thresh")
plt.axis('off')
plt.show()
kernel = np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 20)
plt.imshow(closing,cmap="gray")
plt.title("final noise removal")
plt.axis('off')
plt.show()
kernel = np.ones((5,5),np.uint8)
erd = closing

# sure background area
sure_bg = cv2.dilate(closing,kernel,iterations=5)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(closing,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

plt.imshow(markers)
plt.title("final marked image")
plt.axis('off')
plt.show()


mx = 0
x = 0
y = 0
m = 0
n = 0
p = 0
q = 0
fl = 0
for i in range(len(erd)):
    c = 0
    for j in range(len(erd[0])):
        if erd[i][j] == 0:
            if c==0:
                m = i
                n = j
                q = i
                if p==0:
                    p = i
            c+=1
    if c>mx:
        mx = c
        x = m
        y = n
x = p + (q-p)//2
y = y + mx//2
radius = mx//2
rdX = radius
f = 0
i = 1
j = 0
s = radius
while 1:
    if erd[x + i][y + i] != 0 and erd[x - i][y - i] != 0:
        s += int(1.414 * i)
        break
    else:
        i+=1
while 1:
    if erd[x - i][y + i] != 0 and erd[x+i][y - i] != 0:
        s += int(1.414 * i)
        break
    else:
        i+=1
while 1:
    if erd[x-i][y]!=0 and erd[x+i][y]!=0:
        s+=i
        break
    else:
        i+=1

radius = s//4
print(radius)
window_name = 'erd'

# Center coordinates
center_coordinates = (y, x)

# Radius of circle

# Blue color in BGR
color = (127, 127, 127)

# Line thickness of 2 px
thickness = 5

# Using cv2.circle() method
# Draw a circle with blue line borders of thickness of 2 px
erd = cv2.circle(erd, center_coordinates, rdX, color, thickness)



# Displaying the image
plt.imshow(erd,cmap="gray")
plt.title("final segmented image")
plt.axis('off')
# cv2.imwrite("fnlApprx.jpg",erd)
# plt.savefig("finalSeg.png", transparent = True, bbox_inches = 'tight', pad_inches = 0)
plt.show()
