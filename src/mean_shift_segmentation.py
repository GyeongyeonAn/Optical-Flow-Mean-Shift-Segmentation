from optical_flow import *

# 5-dimension vector
x_vec = np.zeros(5, dtype='float64')
y_vec = np.zeros(5, dtype='float64')
shift_vec = np.zeros(5, dtype='float64')
v = np.zeros((height, width, 5), dtype='float64')

# scale of kernel hs, hr
hs = 5
hr = 5

# parameter min_shift
epsilon = 2
test_width = width / 6
test_hidht = he
for x in range(0, width):
    for y in range(0, height):
        # initial current pixel vector x
        x_vec[0] = y
        x_vec[1] = x
        x_vec[2] = frame_0_img[y, x, 0]
        x_vec[3] = frame_0_img[y, x, 1]
        x_vec[4] = frame_0_img[y, x, 2]

        # initial y
        y_vec = x_vec

        # find of the highest density
        current_vec = np.zeros(5, dtype='float64')
        while True:
            numerator = np.zeros(5, dtype='float64')
            denominator = 0
            for i in range(0, width):
                for j in range(0, height):
                    # calculate distance s, r
                    s_dist = np.sqrt((j - y_vec[0]) ** 2 + (i - y_vec[1]) ** 2)
                    if s_dist < hs:
                        r_dist = np.sqrt((frame_0_img[j, i, 0] - y_vec[2]) ** 2 +
                                         (frame_0_img[j, i, 1] - y_vec[3]) ** 2 +
                                         (frame_0_img[j, i, 2] - y_vec[4]) ** 2)
                        if r_dist < hr:
                            # set of current pixel
                            current_vec[0] = j
                            current_vec[1] = i
                            current_vec[2] = frame_0_img[j, i, 0]
                            current_vec[3] = frame_0_img[j, i, 1]
                            current_vec[4] = frame_0_img[j, i, 2]

                            # find of ks value
                            ks_param = (current_vec - y_vec) / hs
                            l2_norm = np.sqrt(ks_param[0] ** 2 + ks_param[1] ** 2)
                            if l2_norm <= 1:
                                ks = np.exp(-(l2_norm ** 2))
                            else:
                                ks = 0

                            # find of kr value
                            kr_param = (current_vec - y_vec) / hr
                            l2_norm = np.sqrt(kr_param[2] ** 2 + kr_param[3] ** 2 + kr_param[4] ** 2)
                            if l2_norm <= 1:
                                kr = np.exp(-(l2_norm ** 2))
                            else:
                                kr = 0

                            k = ks * kr
                            numerator = numerator + k * current_vec
                            denominator = denominator + k

            # check convergence & save points
            next_y_vec = numerator / denominator
            shift_vec = next_y_vec - y_vec
            l2_norm = np.linalg.norm(shift_vec)
            if l2_norm < epsilon:
                v[y, x] = next_y_vec
                break
            else:
                y_vec = next_y_vec


# clustering vector v
img_size = width * height
assigned_cluster = np.full((height, width), -1, dtype='int64')
c_centroid = []
c_points_num = []
cluster_num = -1
for i in range(img_size):
    w = int(i / height)
    h = int(i % height)
    if assigned_cluster[h, w] == -1:
        cluster_num = cluster_num + 1
        assigned_cluster[h, w] = cluster_num
        c_points_num.append(1)
        c_centroid.append(v[h, w])
        for j in range(i + 1, img_size):
            target_w = int(j / height)
            target_h = int(j % height)
            if assigned_cluster[target_h, target_w] == -1:
                # calculate s, r distance
                s_dist = np.sqrt((v[target_h, target_w, 0] - v[h, w, 0]) ** 2 +
                                 (v[target_h, target_w, 1] - v[h, w, 1]) ** 2)
                if s_dist < hs:
                    r_dist = np.sqrt((v[target_h, target_w, 2] - v[h, w, 2]) ** 2 +
                                     (v[target_h, target_w, 3] - v[h, w, 3]) ** 2 +
                                     (v[target_h, target_w, 4] - v[h, w, 4]) ** 2)
                    if r_dist < hr:
                        assigned_cluster[target_h, target_w] = cluster_num
                        c_points_num[cluster_num] = c_points_num[cluster_num] + 1
                        c_centroid[cluster_num] = c_centroid[cluster_num] + v[target_h, target_w]
        # calculate cluster centroid
        c_centroid[cluster_num] = c_centroid[cluster_num] / c_points_num[cluster_num]

# merge cluster
assigned = np.full(len(c_centroid), -1, dtype='int64')
cluster = []
cluster_count = []
cluster_num = -1
ds = 500
dr = 30
for i in range(len(c_centroid)):
    if assigned[i] == -1:
        cluster_num += 1
        assigned[i] = cluster_num
        cluster_count.append(1)
        cluster.append(c_centroid[i])
        for j in range(i + 1, len(c_centroid)):
            if assigned[j] == -1:
                # calculate s, r distance
                s_dist = np.sqrt((c_centroid[j][0] - c_centroid[i][0]) ** 2 +
                                 (c_centroid[j][1] - c_centroid[i][1]) ** 2)
                if s_dist < ds:
                    r_dist = np.sqrt((c_centroid[j][2] - c_centroid[i][2]) ** 2 +
                                     (c_centroid[j][3] - c_centroid[i][3]) ** 2 +
                                     (c_centroid[j][4] - c_centroid[i][4]) ** 2)
                    if r_dist < dr:
                        assigned[j] = cluster_num
                        cluster_count[cluster_num] = cluster_count[cluster_num] + 1
                        cluster[cluster_num] = cluster[cluster_num] + c_centroid[j]
        # calculate cluster centroid
        cluster[cluster_num] = cluster[cluster_num] / cluster_count[cluster_num]

# express of cluster centroid segmentation
for i in range(width):
    for j in range(height):
        for k in range(3):
            frame_0_img[j, i, k] = cluster[assigned[assigned_cluster[j, i]]][k+2]
# express edge
for i in range(1, width-1):
    for j in range(1, height-1):
        if (assigned[assigned_cluster[j, i]] != assigned[assigned_cluster[j - 1, i]]) or \
                (assigned[assigned_cluster[j, i]] != assigned[assigned_cluster[j, i - 1]]):
            frame_0_img[j, i, 0] = 0
            frame_0_img[j, i, 1] = 0
            frame_0_img[j, i, 2] = 0

# segmentation motion vector
length = len(cluster)
segment_motion = np.zeros((length, 2), dtype='float32')
pixel_num = np.zeros(length, dtype='float32')
for x in range(width):
    for y in range(height):
        number = assigned[assigned_cluster[y, x]]
        segment_motion[number] = segment_motion[number] + motion_vector[y, x]
        pixel_num[number] = pixel_num[number] + 1

for i in range(length):
    center_y = int(cluster[i][0])
    center_x = int(cluster[i][1])
    segment_motion[i] = segment_motion[i] / pixel_num[i]
    dy = int(segment_motion[i][0] * 20)
    dx = int(segment_motion[i][1] * 20)
    cv2.arrowedLine(frame_0_copy_img, (center_x, center_y), (center_x + dx, center_y + dy), (255, 0, 0), 1, tipLength=0.5)

    ax3 = flg.add_subplot(rows, cols, 3)
    ax3.imshow(frame_0_copy_img)
    ax3.set_title("Mean Shift Image")
    ax3.axis('off')
    plt.show()