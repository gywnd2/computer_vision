import cv2, random
import numpy as np
from matplotlib import pyplot as plt

# 영상 불러오기
imgA, imgB=cv2.imread('img/3rd.jpg', cv2.IMREAD_ANYCOLOR), cv2.imread('img/4th.jpg', cv2.IMREAD_ANYCOLOR)
# 영상 A, B에서 클릭 지점, ROI 표시 색상, 히스토그램 연산 결과를 저장할 배열
pointsA, pointsB=[], []
colorA, colorB=[], []
# 패치 크기 / 2
squareSize=50
# 영상 크기 조정
imgA=cv2.resize(imgA, (480, 640))
imgB=cv2.resize(imgB, (480, 640))

# FLANN 기반의 피처 매칭
def sift(roi, image):
    sift=cv2.SIFT_create()

    kp1, des1=sift.detectAndCompute(roi, None)
    kp2, des2=sift.detectAndCompute(image, None)

    FLANN_INDEX_KDTREE=0
    index_params=dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params=dict(check=50)

    flann=cv2.FlannBasedMatcher(index_params, search_params)
    matches=flann.knnMatch(des1, des2, k=2)

    matchesMask=[[0, 0] for i in range(len(matches))]

    for i,(m,n) in enumerate(matches):
        if m.distance<0.7*n.distance:
            matchesMask[i]=[1,0]
            
    draw_params=dict(matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0),
                    matchesMask=matchesMask,
                    flags=0)

    img=cv2.drawMatchesKnn(roi, kp1, image, kp2, matches, None, **draw_params)
    plt.imshow(img,),plt.show()

# 클릭하여 좌표 저장하고 ROI 그리기
def drawROI(event, x, y, flag, param):
    # 영상 A일 경우
    if param=='A':
        # 패치 4개 될 때까지 수행
        if event==cv2.EVENT_LBUTTONUP and len(pointsA)<4:
            # 랜덤 색상 
            colorA.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            pointsA.append((x, y))
            imageA=imgA.copy()
            for i in range(len(pointsA)):
                # 박스와 번호 그리기
                cv2.rectangle(imageA, (pointsA[i][0]-squareSize, pointsA[i][1]+squareSize), (pointsA[i][0]+squareSize, pointsA[i][1]-squareSize), colorA[i], 5)
                cv2.putText(imageA, str(i), (pointsA[i][0]-20, pointsA[i][1]-50), cv2.FONT_HERSHEY_SIMPLEX, 2, colorA[i])
            cv2.imshow("Image A", imageA)
        
        # 4개 저장하고 5번째 클릭 시 히스토그램 4개 계산과 메시지 출력
        elif event==cv2.EVENT_LBUTTONDOWN and len(pointsA)==4:
            print("A에서 4개의 ROI가 저장되었습니다.")
            for i in range(4):
                roi=imgA[pointsA[i][1]-squareSize:pointsA[i][1]+squareSize, pointsA[i][0]-squareSize:pointsA[i][0]+squareSize]
                # 피처 매칭을 위한 패치 영역 영상 정규화
                roiNorm=cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                imgBNorm=cv2.normalize(imgB, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                # 피처 매칭 수행
                sift(roiNorm, imgBNorm)
                
    # 영상 B일 경우
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
            print("B에서 4개의 ROI가 저장되었습니다.")
            for i in range(4):
                roi=imgB[pointsB[i][1]-squareSize:pointsB[i][1]+squareSize, pointsB[i][0]-squareSize:pointsB[i][0]+squareSize]
                roiNorm=cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                imgBNorm=cv2.normalize(imgB, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                sift(roi, imgA)
    
cv2.imshow("Image A", imgA)
cv2.imshow("Image B", imgB)

while True:
    cv2.setMouseCallback("Image A", drawROI, 'A')
    cv2.setMouseCallback("Image B", drawROI, 'B')
    key=cv2.waitKey(0)

    # 영상 A의 ROI 전체 삭제
    if key==ord('a'):
        pointsA.clear()
        cv2.imshow("Image A", imgA)
    # 영상 B의 ROI 전체 삭제
    elif key==ord('b'):
        pointsB.clear()
        cv2.imshow("Image B", imgB)
    # ESC -> 종료
    elif key==27:
        break
    
cv2.destroyAllWindows()