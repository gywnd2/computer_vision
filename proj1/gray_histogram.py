import cv2, random
import numpy as np
from matplotlib import pyplot as plt

# 흑백 영상 불러오기
imgA, imgB=cv2.imread('img/1st.jpg', cv2.IMREAD_GRAYSCALE), cv2.imread('img/2nd.jpg', cv2.IMREAD_GRAYSCALE)
# 영상 A, B에서 클릭 지점, ROI 표시 색상, 히스토그램 연산 결과를 저장할 배열
pointsA, pointsB=[], []
colorA, colorB=[], []
histA, histB=[], []
# 패치 크기 / 2
squareSize=50
# 영상 크기 조정
imgA=cv2.resize(imgA, (480, 640))
imgB=cv2.resize(imgB, (480, 640))

# 히스토그램 비교 함수
def compHists(image):
    for i in range(4):
        res=[]
        for j in range(4):
            # A 영상일 경우 A의 패치와 B의 전체 패치 4개 비교
            if image=='A':
                res.append(cv2.compareHist(histA[i], histB[j], cv2.HISTCMP_CORREL))
            # B 영상일 경우 B의 패치와 A의 전체 패치 4개 비교
            else:
                res.append(cv2.compareHist(histB[i], histA[j], cv2.HISTCMP_CORREL))
        # 터미널에 비교 결과 출력
        print(res)
        # 최종 결과 (가장 유사한 패치 번호) 출력
        if image=='A':
            print('A의 ROI ',i,'번과 가장 유사한 B의 ROI는 ',res.index(max(res)),'번 입니다.')
        else:
            print('B의 ROI ',i,'번과 가장 유사한 A의 ROI는 ',res.index(max(res)),'번 입니다.')
            
# 클릭하여 저장한 ROI의 히스토그램 연산 함수
def calcHistOfROI(image, imgName):
    mask=np.zeros(image.shape[:2], np.uint8)
    # mask[y_min:y_max, x_min:x_max]
    for i in range(4):
        # 영상 A일 경우
        if imgName=='A' and len(histA)!=4:
            # 클릭 지점을 중심으로 한 변의 크기가 squareSize*2인 마스크 생성 하여 히스토그램 계산
            mask[pointsA[i][1]-squareSize:pointsA[i][1]+squareSize, pointsA[i][0]-squareSize:pointsA[i][0]+squareSize]=255
            masked_img=cv2.bitwise_and(image, image, mask=mask)
            hist_full=cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_mask=cv2.calcHist([image], [0], mask, [256], [0, 256])
            histA.append(hist_mask)
        # 영상 B일 경우
        elif imgName=='B' and len(histB)!=4:
            mask[pointsB[i][1]-squareSize:pointsB[i][1]+squareSize, pointsB[i][0]-squareSize:pointsB[i][0]+squareSize]=255
            masked_img=cv2.bitwise_and(image, image, mask=mask)
            
            hist_full=cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_mask=cv2.calcHist([image], [0], mask, [256], [0, 256])
            histB.append(hist_mask)


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
            calcHistOfROI(imgA, "A")
            print("A에서 4개의 ROI가 저장되었습니다.")
        
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
            calcHistOfROI(imgB, "B")
            print("B에서 4개의 ROI가 저장되었습니다.")
    
cv2.imshow("Image A", imgA)
cv2.imshow("Image B", imgB)

while True:
    cv2.setMouseCallback("Image A", drawROI, 'A')
    cv2.setMouseCallback("Image B", drawROI, 'B')
    key=cv2.waitKey(0)

    # 영상 A의 ROI 전체 삭제
    if key==ord('a'):
        pointsA.clear()
        histA.clear()
        cv2.imshow("Image A", imgA)
    # 영상 B의 ROI 전체 삭제
    elif key==ord('b'):
        histB.clear()
        pointsB.clear()
        cv2.imshow("Image B", imgB)
    # 영상 A의 패치와 가장 유사한 B의 패치 찾기
    elif key==ord('c'):
        compHists('A')        
    # 영상 B의 패치와 가장 유사한 A의 패치 찾기
    elif key==ord('v'):
        compHists('B') 
    # ESC -> 종료
    elif key==27:
        break
cv2.destroyAllWindows()