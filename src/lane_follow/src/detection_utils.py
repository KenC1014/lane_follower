import numpy as np
import cv2
from skimage import morphology

def gradient_thresh(img):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        #1. Convert the image to gray scale
        #2. Gaussian blur the image
        #3. Use cv2.Sobel() to find derivatives for both X and Y Axis
        #4. Use cv2.addWeighted() to combine the results
        #5. Convert each pixel to uint8, then apply threshold to get binary image

        ## TODO
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
        sigma = 1
        ksize = 7
        img = cv2.GaussianBlur(img, ksize=(ksize, ksize), sigmaX=sigma)
        img = cv2.medianBlur(img, ksize=3)
        img_edges = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=7)
        # cv2.imwrite("edges.jpg", img_edges)

        min_threshold = 128
        max_threshold = 255
        binary_output = np.zeros_like(img).astype(np.uint8)
        binary_output[(min_threshold <= img_edges) & (img_edges < max_threshold)] = 1
        ####
        # cv2.imwrite("post_gradient.jpg", binary_output * 255)
        return binary_output


def color_thresh(img):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        #1. Convert the image from RGB to HSL
        #2. Apply threshold on S channel to get binary image
        #Hint: threshold on H to remove green grass
        ## TODO
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(img)
        min_s, max_s = (100, 255)
        binary_output = np.zeros_like(s).astype(np.uint8)
        binary_output[(min_s <= s) & (s <= max_s)] = 1
        min_h, max_h = (15, 85)
        mask_h = np.zeros_like(h).astype(np.uint8)
        mask_h[(min_h <= h) & (h <= max_h)] = 1
        binary_output[mask_h == 1] = 0
        ####

        # cv2.imwrite("post_color.jpg", binary_output * 255)
        return binary_output


def combinedBinaryImage(img):
        """
        Get combined binary image from color filter and sobel filter
        """
        #1. Apply sobel filter and color filter on input image
        #2. Combine the outputs
        ## Here you can use as many methods as you want.
        # cv2.imwrite("test_raw.jpg", img)
        ## TODO

        # Thesholds for real time
        min_thresh = 100
        max_thresh = 200

        # Thesholds for simulation
        # min_thresh = 17
        # max_thresh = 50

        binaryImage = cv2.Canny(img, min_thresh, max_thresh)
        # cv2.imwrite("combine.jpg", binaryImage)

        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)

        return binaryImage

def perspective_transform(img, mode="front", verbose=False):
        """
        Get bird's eye view from input image
        """
        #1. Visually determine 4 source points and 4 destination points
        #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        #3. Generate warped image in bird view using cv2.warpPerspective()

        ## TODO
        img = img.astype(np.uint8)
        h, w = img.shape
        print(h,w)

        # Transform for real time
        x_tl = 640 - 66
        x_tr = w - x_tl
        y_t = 403
        x_bl = 640 - 340
        x_br = w - x_bl
        y_b = h
        x_l_trans = x_bl
        x_r_trans = x_br
        y_t_trans = 0
        y_b_trans = h

        # Transform for GEM simulation
        # x_tl = 280
        # x_tr = 358
        # y_t = 260
        # x_bl = 2
        # x_br = 615
        # y_b = 405
        # x_l_trans = x_bl
        # x_r_trans = x_br
        # y_t_trans = 0
        # y_b_trans = 405

        # # left camera view
        # if mode == "left":
        #         x_tl = 243
        #         x_tr = 657
        #         y_t = 530
        #         x_bl = 734
        #         x_br = 1088
        #         y_b = 690
        #         x_l_trans = x_bl
        #         x_r_trans = x_br
        #         y_t_trans = 0
        #         y_b_trans = y_b


        # # right camera view
        # elif mode == "right":
        #         x_tl = 517
        #         x_tr = 760
        #         y_t = 427
        #         x_bl = 302
        #         x_br = 1050
        #         y_b = 660
        #         x_l_trans = x_bl
        #         x_r_trans = x_br
        #         y_t_trans = 0
        #         y_b_trans = y_b
                
        camera_points = np.array([[x_tl, y_t], [x_tr, y_t],[x_br, y_b], [x_bl, y_b]], dtype=np.float32)
        birdeye_points = np.array([[x_l_trans, y_t_trans], [x_r_trans, y_t_trans],
                                [x_r_trans, y_b_trans], [x_l_trans, y_b_trans]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(camera_points, birdeye_points)
        Minv = cv2.getPerspectiveTransform(birdeye_points, camera_points)

        # cv2.imwrite("test_prebird.jpg", img * 255)
        warped_img = cv2.warpPerspective(img, M, dsize=(w, h))
        # cv2.imwrite("test_postbird.jpg", warped_img * 255)
        ####

        return warped_img, M, Minv
