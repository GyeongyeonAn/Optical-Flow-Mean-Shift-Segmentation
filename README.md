# Optical Flow & Mean Shift Segmentation
Optical Flow와 Mean Shift Segmentation 하는 기법을 학습하는 목적으로 진행한다.

### 개요
-  움직임이 존재하는 2장의 연속 (t, t+1) 프레임 (RGB 영상)을 입력으로 사용한다.
-  Optical flow를 이용하여 앞쪽 영상 (t) 내 픽셀의 motion vector (x,y) 을 도출한다.
-  K-means clustering을 t번째 영상에 적용하여 segmentation 결과 도출, 다만 RGB 값 이외에 motion vector도 함께 고려할 수 있도록 한다.
-  위와 같은 상황에서 mean-shift 알고리즘을 적용하여 segmentation 결과 도출

### 1. 개발 환경

### 2. 코드 설명
#### A. Flow Chart
#### B. Detail Process

### 3. 정확도 평가
