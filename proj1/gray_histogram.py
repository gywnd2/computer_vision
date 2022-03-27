import cv2, random
import numpy as np
from matplotlib import pyplot as plt

imgA, imgB=cv2.imread('img/1st.jpg', cv2.IMREAD_GRAYSCALE), cv2.imread('img/2nd.jpg', cv2.IMREAD_GRAYSCALE)
pointsA, pointsB=[], []
colorA, colorB=[], []
histA, histB=[], []
squareSize=50
imgA=cv2.resize(imgA, (640, 480))
imgB=cv2.resize(imgB, (640, 480))

def compHists(image):
    for i in range(4):
        res=[]
        for j in range(4):
            if image=='A':
                res.append(cv2.compareHist(histA[i], histB[j], cv2.HISTCMP_CORREL))
            else:
                res.append(cv2.compareHist(histB[i], histA[j], cv2.HISTCMP_CORREL))
        print(res)
        if image=='A':
            print('A의 ROI ',i,'번과 가장 유사한 B의 ROI는 ',res.index(max(res)),'번 입니다.')
        else:
            print('B의 ROI ',i,'번과 가장 유사한 A의 ROI는 ',res.index(max(res)),'번 입니다.')
            
def calcHistOfROI(image, imgName):
    mask=np.zeros(image.shape[:2], np.uint8)
    # mask[y_min:y_max, x_min:x_max]
    for i in range(4):
        if imgName=='A' and len(histA)!=4:
            mask[pointsA[i][1]-squareSize:pointsA[i][1]+squareSize, pointsA[i][0]-squareSize:pointsA[i][0]+squareSize]=255
            masked_img=cv2.bitwise_and(image, image, mask=mask)
            hist_full=cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_mask=cv2.calcHist([image], [0], mask, [256], [0, 256])
            histA.append(hist_mask)
        elif imgName=='B' and len(histB)!=4:
            mask[pointsB[i][1]-squareSize:pointsB[i][1]+squareSize, pointsB[i][0]-squareSize:pointsB[i][0]+squareSize]=255
            masked_img=cv2.bitwise_and(image, image, mask=mask)
            
            hist_full=cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_mask=cv2.calcHist([image], [0], mask, [256], [0, 256])
            histB.append(hist_mask)

def drawROI(event, x, y, flag, param):
    if param=='A':
        if event==cv2.EVENT_LBUTTONUP and len(pointsA)<4:
            colorA.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            pointsA.append((x, y))
            imageA=imgA.copy()
            for i in range(len(pointsA)):
                cv2.rectangle(imageA, (pointsA[i][0]-squareSize, pointsA[i][1]+squareSize), (pointsA[i][0]+squareSize, pointsA[i][1]-squareSize), colorA[i], 5)
                cv2.putText(imageA, str(i), (pointsA[i][0]-20, pointsA[i][1]-50), cv2.FONT_HERSHEY_SIMPLEX, 2, colorA[i])
            cv2.imshow("Image A", imageA)
        
        elif event==cv2.EVENT_LBUTTONDOWN and len(pointsA)==4:
            calcHistOfROI(imgA, "A")
            print("A에서 4개의 ROI가 저장되었습니다.")
        
    else:
        if event==cv2.EVENT_LBUTTONUP and len(pointsB)<4:
            colorB.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            pointsB.append((x, y))
            imageB=imgB.copy()
            for i in range(len(pointsB)):
                cv2.rectangle(imageB, (pointsB[i][0]-squareSize, pointsB[i][1]+squareSize), (pointsB[i][0]+squareSize, pointsB[i][1]-squareSize), colorB[i], 5)
                cv2.putText(imageB, str(i), (pointsB[i][0]-20, pointsB[i][1]-50), cv2.FONT_HERSHEY_SIMPLEX, 2, colorB[i])
            cv2.imshow("Image B", imageB) 
        elif event==cv2.EVENT_LBUTTONDOWN and len(pointsB)==4:
            calcHistOfROI(imgB, "B")
            print("B에서 4개의 ROI가 저장되었습니다.")
    
cv2.imshow("Image A", imgA)
cv2.imshow("Image B", imgB)
while True:
    cv2.setMouseCallback("Image A", drawROI, 'A')
    cv2.setMouseCallback("Image B", drawROI, 'B')
    key=cv2.waitKey(0)

    if key==ord('a'):
        pointsA.clear()
        histA.clear()
        cv2.imshow("Image A", imgA)
    elif key==ord('b'):
        histB.clear()
        pointsB.clear()
        cv2.imshow("Image B", imgB)
    elif key==ord('c'):
        print(len(histA), len(histB))
        compHists('A')        
    elif key==ord('v'):
        print(len(histA), len(histB))
        compHists('B') 
    elif key==27:
        break
cv2.destroyAllWindows()