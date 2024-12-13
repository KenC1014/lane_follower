import numpy as np
import cv2
from skimage import morphology
import skimage


def color_thresh(img):
    # img = cv2.medianBlur(img, 5)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(img)
    binary_output = np.zeros_like(s).astype(np.uint8)

    # min_h, max_h = (0, 180)
    # mask_h = np.zeros_like(binary_output).astype(np.uint8)
    # mask_h[(min_h <= h) & (h <= max_h)] = 1
    min_h, max_h = (40, 80)
    mask_h = np.ones_like(binary_output).astype(np.uint8)
    # mask_h[(min_h <= h) & (h <= max_h)] = 0

    min_l, max_l = (128, 255)
    mask_l = np.zeros_like(binary_output).astype(np.uint8)
    # mask_l[(min_l <= l) & (l <= max_l)] = 1
    mask_l[l >= np.percentile(l, 25)] = 1

    min_s, max_s = (128, 255)
    mask_s = np.ones_like(binary_output).astype(np.uint8)
    # mask_s[(min_s <= s) & (s <= max_s)] = 0

    binary_output[(mask_h == 1) & (mask_l == 1) & (mask_s == 1)] = 1

    return binary_output


def combinedBinaryImage(img):
    color_mask = color_thresh(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    # img = cv2.medianBlur(img, 5)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    # img = cv2.bilateralFilter(img, 9, 75, 75)

    binaryImage = cv2.Canny(img, 50, 150)
    # binaryImage[color_mask == 0] = 0

    # filter out small edges
    # contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filtered_edges = np.zeros_like(binaryImage)
    # for contour in contours:
    #     # Calculate the area of the contour
    #     area = cv2.contourArea(contour)
    #     # Retain only contours larger than the specified minimum size
    #     if area >= 100:
    #         cv2.drawContours(filtered_edges, [contour], -1, 255, thickness=cv2.FILLED)

    # Remove noise from binary image
    binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=100,connectivity=2)

    cv2.imwrite("color_mask.jpg", np.hstack((color_mask, binaryImage)) * 255)

    return binaryImage


# for front camera only
def perspective_transform(img, model="e4"):
    """
    Get bird's eye view from input image
    """
    # 1. Visually determine 4 source points and 4 destination points
    # 2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
    # 3. Generate warped image in bird view using cv2.warpPerspective()

    ## TODO
    img = img.astype(np.uint8)
    h, w = img.shape

    ############### e4 ####################
    x_tl = w // 2 - 60  # (524 - 643) // 2   # half view distance: 640 + (584 - 647) // 2
    x_tr = w - x_tl
    y_t = 420
    x_bl = w // 2 - 300  # (277 - 876) // 2
    x_br = w - x_bl
    y_b = h
    x_l_trans = x_bl
    x_r_trans = x_br
    y_t_trans = 0
    y_b_trans = h * 3.9

    ############### ee ####################
    if model == "e2":
        x_tl = w // 2 - 66  # (552 - 690) // 2   # half view distance: 640 + (584 - 647) // 2
        x_tr = w - x_tl
        y_t = 403
        x_bl = w // 2 - 340  # (314 - 994) // 2
        x_br = w - x_bl
        y_b = h
        x_l_trans = x_bl
        x_r_trans = x_br
        y_t_trans = 0
        y_b_trans = h * 2.92

    camera_points = np.array([[x_tl, y_t], [x_tr, y_t], [x_br, y_b], [x_bl, y_b]], dtype=np.float32)
    birdeye_points = np.array([[x_l_trans, y_t_trans], [x_r_trans, y_t_trans],
                               [x_r_trans, y_b_trans], [x_l_trans, y_b_trans]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(camera_points, birdeye_points)
    Minv = cv2.getPerspectiveTransform(birdeye_points, camera_points)

    # cv2.imwrite("src/lane_follow/src/test_prebird.jpg", img * 255)
    warped_img = cv2.warpPerspective(img, M, dsize=(w, h))
    # cv2.imwrite("src/lane_follow/src/test_postbird.jpg", warped_img * 255)
    ####

    return warped_img, M, Minv


# for three view
def get_perspective_points(img, out_h, out_w, scale, mode="e4front"):
    img = img.astype(np.uint8)
    img_h, img_w = img.shape[:2]
    vertical_shift = 0

    ############### e4 ####################
    if "e4" in mode:
        vertical_shift = -100
    # mode == "e4front"
    x_tl = img_w // 2 - 60  # (524 - 643) // 2   # half view distance: 640 + (584 - 647) // 2
    x_tr = img_w - x_tl
    y_t = 420
    x_bl = img_w // 2 - 300  # (277 - 876) // 2
    x_br = img_w - x_bl
    y_b = img_h
    x_l_trans = x_bl + (out_w - img_w) // 2
    x_r_trans = x_br + (out_w - img_w) // 2
    y_t_trans = 0
    y_b_trans = img_h * 3.9

    trans_to_zero = 0
    theta = 0
    translation = np.array([0, vertical_shift])
    adjust_size = 1

    # left camera view
    if mode == "e4left":
        x_tl = img_w // 2 - 50 # 78  # 217 # (1004 - 1159) // 2
        x_tr = img_w - x_tl
        y_t = 400
        x_bl = img_w // 2 - 362  # 599 # （747 - 1470) // 2
        x_br = img_w - x_bl
        y_b = 800
        x_l_trans = x_bl
        x_r_trans = x_br
        y_t_trans = y_t
        y_b_trans = y_b * 8.4
        trans_to_zero = np.array([1 - img_w, 1 - img_h])
        theta = -47 * np.pi / 180
        translation = np.array([(out_w - img_w) // 2 -3229, vertical_shift -1501])
        adjust_size = 1

    # right camera view
    elif mode == "e4right":
        x_tl = img_w // 2 - 42  # 56  # (675 - 787) // 2
        x_tr = img_w - x_tl
        y_t = 400
        x_bl = img_w // 2 - 342  # (414 - 1097) // 2
        x_br = img_w - x_bl
        y_b = 800
        x_l_trans = x_bl
        x_r_trans = x_br
        y_t_trans = y_t
        y_b_trans = y_b * 7.4 # 8.24
        trans_to_zero = np.array([0, 1 - img_h])
        theta = 55 * np.pi / 180
        translation = np.array([(out_w - img_w) // 2 + 5923, vertical_shift -1188])
        adjust_size = 1.19

    ############ e2 ##################
    if "e2" in mode:
        vertical_shift = 0
    # front camera view
    elif mode == "e2front":
        x_tl = img_w // 2 - 66  # (552 - 690) // 2   # half view distance: 640 + (584 - 647) // 2
        x_tr = img_w - x_tl
        y_t = 403
        x_bl = img_w // 2 - 340  # (314 - 994) // 2
        x_br = img_w - x_bl
        y_b = img_h
        x_l_trans = x_bl + (out_w - img_w) // 2
        x_r_trans = x_br + (out_w - img_w) // 2
        y_t_trans = 0
        y_b_trans = img_h * 2.92
        trans_to_zero = 0
        theta = 0
        translation = np.array([0, vertical_shift])
        adjust_size = 1

    # left camera view
    elif mode == "e2left":
        x_tl = img_w // 2 - 210  # 217 # (70 - 505) // 2
        x_tr = img_w - x_tl
        y_t = 400
        x_bl = img_w // 2 - 600  # 599 # （372 - 1571) // 2
        x_br = img_w - x_bl
        y_b = 800
        x_l_trans = x_bl
        x_r_trans = x_br
        y_t_trans = y_t
        y_b_trans = y_b * 3
        trans_to_zero = np.array([1 - img_w, 1 - img_h])
        theta = -48 * np.pi / 180
        translation = np.array([(out_w - img_w) // 2 + 657, vertical_shift + 990])
        adjust_size = 0.75

    # right camera view
    elif mode == "e2right":
        x_tl = img_w // 2 - 165  # 392
        x_tr = img_w - x_tl
        y_t = 400
        x_bl = img_w // 2 - 600  # 567  # （433 - 1567) // 2
        x_br = img_w - x_bl
        y_b = 800
        x_l_trans = x_bl
        x_r_trans = x_br
        y_t_trans = y_t
        y_b_trans = y_b * 3.2
        trans_to_zero = np.array([0, 1 - img_h])
        theta = 43 * np.pi / 180
        translation = np.array([(out_w - img_w) // 2 + 1510, vertical_shift + 1270])
        adjust_size = 0.72

    camera_points = np.array([[x_tl, y_t], [x_tr, y_t], [x_br, y_b], [x_bl, y_b]], dtype=np.float32)
    birdeye_points = np.array([[x_l_trans, y_t_trans], [x_r_trans, y_t_trans],
                               [x_r_trans, y_b_trans], [x_l_trans, y_b_trans]], dtype=np.float32)
    birdeye_points *= adjust_size
    if "left" in mode or "right" in mode:
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        birdeye_points += trans_to_zero
        birdeye_points = (rotation @ birdeye_points.reshape([4, 2, 1])).reshape([4, 2])
    birdeye_points = (birdeye_points + translation).astype(np.float32)
    birdeye_points *= scale

    return camera_points, birdeye_points


def warp_image(image, out_h, out_w, scale, mode="front"):
    image = image.astype(np.uint8)
    camera_points, birdeye_points = get_perspective_points(image, out_h, out_w, scale, mode=mode)
    M = cv2.getPerspectiveTransform(camera_points, birdeye_points)
    M_inv = cv2.getPerspectiveTransform(birdeye_points, camera_points)
    out_h = int(out_h * scale)
    out_w = int(out_w * scale)
    warped_img = cv2.warpPerspective(image, M, dsize=(out_w, out_h))
    return warped_img, M, M_inv


# main function
def get_three_view_birdeye(front_image, left_image, right_image, output_h, output_w, scale=1, model="e4"):
    warped_front, M, M_inv = warp_image(front_image,output_h, output_w, scale, mode=model+"front")
    warped_left, _, _ = warp_image(left_image, output_h, output_w, scale, mode=model+"left")
    warped_right, _, _ = warp_image(right_image, output_h, output_w, scale, mode=model+"right")

    # cv2.imwrite("metric_front.jpg", warped_front)
    # cv2.imwrite("metric_left.jpg", warped_left)
    # cv2.imwrite("metric_right.jpg", warped_right)

    warped_image = np.where(warped_front == 0, warped_left, warped_front)
    warped_image = np.where(warped_image == 0, warped_right, warped_image)
    # warped_image = warped_front
    # warped_image = np.where(warped_image == 0, warped_left,
    #                         np.where(warped_left == 0, warped_image, warped_image / 2 + warped_left / 2))
    # warped_image = np.where(warped_image == 0, warped_right,
    #                         np.where(warped_right == 0, warped_image, warped_image / 2 + warped_right / 2))

    # cv2.imwrite('three_view_bird.jpg', warped_image)
    return warped_image, M, M_inv


def get_warped_mask(image, output_h, output_w, shrink, scale, mode):
    warped_mask = np.ones(image.shape[:2])
    warped_mask[:shrink, :] = 0
    warped_mask[-shrink:, :] = 0
    warped_mask[:, -shrink:] = 0
    warped_mask[:, :shrink] = 0
    warped_mask, _, _ = warp_image(warped_mask, output_h, output_w, scale, mode=mode)
    return warped_mask

def get_three_view_birdeye_trans_first(front_image, left_image, right_image, output_h, output_w, scale=1, model="e4"):
    shrink = int(2 / scale)

    warped_left, M_left, Minv_left = warp_image(left_image, output_h, output_w, scale, mode=model+"left")
    cv2.imwrite("warped_left.jpg", warped_left)
    mask_left = get_warped_mask(left_image, output_h, output_w, shrink, scale, mode=model+"left")
    mask_left[:, (mask_left.shape[1] // 2):] = 0
    edges_left = combinedBinaryImage(warped_left)
    edges_left = np.where(mask_left == 1, edges_left, 0)
    cv2.imwrite("warped_left_edges.jpg", edges_left * 255)

    warped_right, M_right, Minv_right = warp_image(right_image, output_h, output_w, scale, mode=model+"right")
    cv2.imwrite("warped_right.jpg", warped_right)
    mask_right = get_warped_mask(right_image, output_h, output_w, shrink, scale, mode=model + "right")
    mask_right[:, :(mask_right.shape[1] // 2)] = 0
    edges_right = combinedBinaryImage(warped_right)
    edges_right= np.where(mask_right == 1, edges_right, 0)
    cv2.imwrite("warped_right_edges.jpg", edges_right * 255)

    front_image = cv2.GaussianBlur(front_image,(5, 5), 0)
    warped_front, M_front, Minv_front = warp_image(front_image, output_h, output_w, scale, mode=model+"front")
    cv2.imwrite("warped_front.jpg", warped_front)
    mask_front = get_warped_mask(front_image, output_h, output_w, shrink, scale, mode=model + "front")
    edges_front = combinedBinaryImage(warped_front)
    edges_front= np.where(mask_front == 1, edges_front, 0)
    cv2.imwrite("warped_front_edges.jpg", edges_front * 255)
    mask_h, mask_w = mask_front.shape[:2]
    mask_front[:,int(mask_w * 5 / 12 ):int(mask_w * 7 / 12 )] = 1   # cut the tire edges

    edges_image = edges_left + edges_right
    warped_edges = np.where(mask_front == 0, edges_image, edges_front)
    cv2.imwrite("warped_edges.jpg", warped_edges * 255)

    return warped_edges, M_front, Minv_front



def get_three_view_forward(birdeye_image, front_h, front_w, Minv_front, out_size=None):
    # warped_image = cv2.warpPerspective(birdeye_image, M_inv, dsize=(front_w, front_h))
    out_h, out_w = front_h, front_w
    if out_size is not None:
        out_h, out_w = out_size
    transform = skimage.transform.ProjectiveTransform(Minv_front)
    offset = (out_w - front_w) // 2, 0 # - front_h // 2
    translation = skimage.transform.SimilarityTransform(translation=offset)
    warped_image = skimage.transform.warp(birdeye_image, (transform + translation).inverse, output_shape=(out_h, out_w))

    # cv2.imwrite('three_view_forward.jpg', warped_image)
    return warped_image


# def get_three_view_forward(front_image, left_image, right_image, output_h, output_w, scale=1, model="e4"):
#     camera_points, birdeye_points = get_perspective_points(front_image, output_h, output_w, scale=scale, mode=model+"front")
#     M_front = cv2.getPerspectiveTransform(camera_points, birdeye_points)
#     Minv_front = cv2.getPerspectiveTransform(birdeye_points, camera_points)

#     camera_points, birdeye_points = get_perspective_points(left_image, output_h, output_w, scale=scale, mode=model+"left")
#     M_left = cv2.getPerspectiveTransform(camera_points, birdeye_points)
#     Minv_left= cv2.getPerspectiveTransform(birdeye_points, camera_points)

#     camera_points, birdeye_points = get_perspective_points(right_image, output_h, output_w, scale=scale, mode=model+"right")
#     M_right = cv2.getPerspectiveTransform(camera_points, birdeye_points)
#     Minv_right = cv2.getPerspectiveTransform(birdeye_points, camera_points)

#     transform_left = skimage.transform.ProjectiveTransform(Minv_front @ M_left)
#     transform_right = skimage.transform.ProjectiveTransform(Minv_front @ M_right)
#     offset = (output_w - front_image.shape[1]) // 2, 0  # - front_image.shape[0] // 2
#     translation = skimage.transform.SimilarityTransform(translation=offset)

#     warped_front = skimage.transform.warp(front_image, (translation).inverse, output_shape=(output_h, output_w))
#     warped_left = skimage.transform.warp(left_image, (transform_left).inverse, output_shape=(output_h, output_w))
#     warped_right = skimage.transform.warp(right_image, (transform_right).inverse, output_shape=(output_h, output_w))

#     warped_image = warped_front
#     warped_image = np.where(warped_front == 0, warped_left, warped_front)
#     warped_image = np.where(warped_image == 0, warped_right, warped_image)

#     return warped_image
