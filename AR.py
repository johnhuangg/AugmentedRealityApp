import cv2
import numpy as np

cap= cv2.VideoCapture(0)
imageTarget = cv2.imread('image.jpg')
vid=cv2.VideoCapture('video.mp4')

detection= False
frameCounter=0

success, imageVideo=vid.read()
height, width, channel=imageTarget.shape
imageVideo=cv2.resize(imageVideo, (width,height))

orb = cv2.ORB_create(nfeatures=1000)
keypt1, desc1=orb.detectAndCompute(imageTarget,None)
#imageTarget = cv2.drawKeypoints(imageTarget,keypt1, None)

def stackImages(imgArray,scale,lables=[]):
    sizeW= imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW,sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((sizeH, sizeW, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

while True:


    success, imageWebcam = cap.read()
    imageAugment=imageWebcam.copy()
    keypt2, desc2 = orb.detectAndCompute(imageWebcam, None)
    #imageWebcam = cv2.drawKeypoints(imageWebcam, keypt2, None)

    if detection== False:
        vid.set(cv2.CAP_PROP_POS_FRAMES,0)
        frameCounter=0
    else:
        if frameCounter == vid.get(cv2.CAP_PROP_FRAME_COUNT):
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success,imageVideo = vid.read()
        imageVideo = cv2.resize(imageVideo, (width, height))

    bf=cv2.BFMatcher()
    matches = bf.knnMatch(desc1,desc2,k=2)
    good=[]

    for m,n in matches:
        if m.distance < 0.75 *n.distance:
            good.append(m)
    print(len(good))

    imageFeatures=cv2.drawMatches(imageTarget,keypt1,imageWebcam,keypt2,good,None,flags=2)

    if len(good) > 20:
        detection=True
        srcPts = np.float32([keypt1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([keypt2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix,mask =cv2.findHomography(srcPts,dstPts,cv2.RANSAC, 5)
        print (matrix)
        pts=np.float32([[0,0],[0,height],[width,height],[width,0]]).reshape(-1,1,2)
        dst=cv2.perspectiveTransform(pts,matrix)
        img2=cv2.polylines(imageWebcam, [np.int32(dst)],True,(255,0,255),3)

        imageWarp=cv2.warpPerspective(imageVideo,matrix, (imageWebcam.shape[1],imageWebcam.shape[0]))
        maskNew=np.zeros((imageWebcam.shape[0],imageWebcam.shape[1]),np.uint8)
        cv2.fillPoly(maskNew,[np.int32(dst)],(255,255,255))
        maskInv = cv2.bitwase_not(maskNew)
        imageAugment=cv2.bitwise_and(imageAugment,imageAugment,maskInv)
        imageAugment=cv2.bitwise_or(imageWarp,imageAugment)
        imageStacked=stackImages(([imageWebcam,imageVideo,imageTarget],[imageFeatures,imageWarp,imageAugment]),0.5)

    # cv2.imshow('maskNew', maskNew)
    # cv2.imshow('imgWarp', imageWarp)
    # cv2.imshow('img2',img2)
    # cv2.imshow('ImageFeatures', imageFeatures)
    # cv2.imshow('ImageTarget',imageTarget)
    # cv2.imshow('vid',imageVideo)
    cv2.imshow('stacked', imageStacked)
    cv2.waitKey(1)
    frameCounter+=1