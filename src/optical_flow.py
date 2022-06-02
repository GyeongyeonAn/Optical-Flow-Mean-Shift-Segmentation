import cv2
import matplotlib.pyplot as plt
import numpy as np

frame_0_img = cv2.imread('../data/image/unscreen-001.png', cv2.IMREAD_COLOR)
frame_1_img = cv2.imread('../data/image/unscreen-150.png', cv2.IMREAD_COLOR)

resize_for_frame_0_img = cv2.imread('../data/image/unscreen-001.png', cv2.IMREAD_COLOR)

frame_0 = cv2.cvtColor(frame_0_img, cv2.COLOR_BGR2GRAY)
frame_1 = cv2.cvtColor(frame_1_img, cv2.COLOR_BGR2GRAY)

height, width = frame_0.shape;
dx = np.zeros(frame_0.shape, dtype='float64')
dy = np.zeros(frame_0.shape, dtype='float64')
dt = np.zeros(frame_0.shape, dtype='float64')

# calculate dx, dy, dt
for x in range(width):
    for y in range(height):
        if x + 1 == width and y + 1 == height:
            dx[y, x] = 0 - frame_0[y, x]
            dy[y, x] = 0 - frame_0[y, x]
        elif x + 1 == width:
            dx[y, x] = 0 - frame_0[y, x]
            dy[y, x] = float(frame_0[y + 1, x]) - float(frame_0[y, x])
        elif y + 1 == height:
            dx[y, x] = float(frame_0[y, x + 1]) - float(frame_0[y, x])
            dy[y, x] = 0 - frame_0[y, x]
        else:
            dx[y, x] = float(frame_0[y, x + 1]) - float(frame_0[y, x])
            dy[y, x] = float(frame_0[y + 1, x]) - float(frame_0[y, x])
        dt[y, x] = float(frame_1[y, x]) - float(frame_0[y, x])

# calculate motion vector
window_size = 13
border = int(window_size / 2)
A = np.zeros((window_size ** 2, 2), dtype='float64')
b = np.zeros((window_size ** 2, 1), dtype='float64')
motion_vector = np.zeros((height, width, 2), dtype='float64')
for x in range(border, width-border):
    for y in range(border, height-border):
        # window 구성
        # 행렬 A, b 생성
        idx = 0
        for i in range(-border, border+1):
            for j in range(-border, border+1):
                A[idx, 0] = dy[y+i, x+j]
                A[idx, 1] = dx[y+i, x+j]
                b[idx] = -dt[y+i, x+j]
                idx += 1

        # motion vector 계산
        result = A.T
        result = result.dot(A)
        det = result[0, 0] * result[1, 1] - result[0, 1] * result[1, 0]
        if det != 0:
            result = np.linalg.inv(result)
        else:
            result = np.linalg.pinv(result)
        result = result.dot(A.T)
        result = result.dot(b)
        motion_vector[y, x] = result.T

for x in range(3, width-3):
    for y in range(3, height-3):
        if x % 7 == 3 and y % 7 == 3:
            vector_x = 0
            vector_y = 0
            # 7x7 윈도우 내 벡터의 합
            for i in range(-3, 4):
                for j in range(-3, 4):
                    vector_y = vector_y + motion_vector[y + i, x + j, 0]
                    vector_x = vector_x + motion_vector[y + i, x + j, 1]
            if vector_y != 0 or vector_x != 0:
                # normalization
                arrow_size = 5
                norm = np.sqrt(vector_y ** 2 + vector_x ** 2)
                norm_y = int(round(arrow_size * (vector_y / norm)))
                norm_x = int(round(arrow_size * (vector_x / norm)))
                if norm_y > 0 or norm_x > 0:
                    cv2.arrowedLine(frame_0_img, (x, y), (x + norm_x, y + norm_y), (255, 0, 0), 1, tipLength=0.5)
                # resize
                resize = 15
                iy = int(round((vector_y / 49) * resize))
                ix = int(round((vector_x / 49) * resize))
                if iy > 0 or iy > 0:
                    cv2.arrowedLine(resize_for_frame_0_img, (x, y), (x + ix, y + iy), (255, 0, 0), 1, tipLength=0.3)

flg = plt.figure()
rows = 1
cols = 2
ax1 = flg.add_subplot(rows, cols, 1)
ax1.imshow(frame_0_img)
ax1.set_title('Normalization Image')
ax1.axis('off')
ax2 = flg.add_subplot(rows, cols, 2)
ax2.imshow(resize_for_frame_0_img)
ax2.set_title("Resize Image")
ax2.axis('off')
plt.show()


