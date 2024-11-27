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
        min_threshold = 128
        max_threshold = 255
        binary_output = np.zeros_like(img).astype(np.uint8)
        binary_output[(min_threshold <= img_edges) & (img_edges < max_threshold)] = 1
        ####

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
        SobelOutput = gradient_thresh(img)
        ColorOutput = color_thresh(img)
        ####

        binaryImage = np.zeros_like(SobelOutput)
        binaryImage[(ColorOutput==1)|(SobelOutput==1)] = 1
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

        # front camera view
        transform_points = {"x_tl": 517,
                            "x_tr": 760,
                            "y_t": 427,
                            "x_bl": 302,
                            "x_br": 1050,
                            "y_b": 660,
                            "y_t_trans": 0,
                            "y_b_shift": 0
                            }
        
        # left camera view
        if mode == "left":
            transform_points = {"x_tl": 1667,
                            "x_tr": 1872,
                            "y_t": 574,
                            "x_bl": 1104,
                            "x_br": 1502,
                            "y_b": 1072,
                            "y_t_trans": 0,
                            "y_b_shift": 0
                            }
        # right camera view
        elif mode == "right":
           transform_points = {"x_tl": 28,
                            "x_tr": 246,
                            "y_t": 463,
                            "x_bl": 135,
                            "x_br": 853,
                            "y_b": 982,
                            "y_t_trans": 0,
                            "y_b_shift": 0
                            }
        
        x_tl = transform_points["x_tl"]
        x_tr = transform_points["x_tr"]
        y_t = transform_points["y_t"]
        x_bl = transform_points["x_bl"]
        x_br = transform_points["x_br"]
        y_b = transform_points["y_b"]
        x_l_trans = 100
        x_r_trans = w - 100
        y_t_trans = transform_points["y_t_trans"]
        y_b_shift = transform_points["y_b_shift"]
        if y_b + y_b_shift > h:
            y_b_shift = h - y_b
        y_b_trans = y_b + y_b_shift
        
        camera_points = np.array([[x_tl, y_t], [x_tr, y_t],[x_br, y_b], [x_bl, y_b]], dtype=np.float32)
        birdeye_points = np.array([[x_l_trans, y_t_trans], [x_r_trans, y_t_trans],
                                   [x_r_trans, y_b_trans], [x_l_trans, y_b_trans]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(camera_points, birdeye_points)
        Minv = cv2.getPerspectiveTransform(birdeye_points, camera_points)

        #cv2.imwrite("test_prebird.jpg", img * 255)
        warped_img = cv2.warpPerspective(img, M, dsize=(w, h))
        #cv2.imwrite("test_postbird.jpg", warped_img * 255)
        ####

        return warped_img, M, Minv
