import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
# from combined_thresh import combined_thresh
# from perspective_transform import perspective_transform

# feel free to adjust the parameters in the code if necessary

def calculate_slope(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	if x2 == x1:
		return 100
	return (y2 - y1) / (x2 - x1)

def line_fit(binary_warped, left_start=0, left_end=None, right_start=None, right_end=None):
	"""
	Find and fit lane lines
	"""
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	height, width = binary_warped.shape

	histogram = np.sum(binary_warped[height//2:,:], axis=0)
	# Create an output image to draw on and visualize the result
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	
	midpoint = int(width/2)
	if left_end == None:
		left_end = midpoint
	if right_start == None:
		right_start = midpoint
	if right_end == None:
		right_end = width
	
	leftx_base = np.argmax(histogram[left_start:left_end]) + left_start
	rightx_base = np.argmax(histogram[right_start:right_end]) + right_start

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = int(height/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		##TO DO
		window_center = (leftx_current + rightx_current) // 2
		window_x_left_min = leftx_current - margin if leftx_current - margin >= 0 else 0
		window_x_left_max = leftx_current + margin if leftx_current + margin <= window_center else window_center
		window_x_right_min = rightx_current - margin if rightx_current - margin >= window_center else window_center
		window_x_right_max = rightx_current + margin if rightx_current + margin <= width else width
		window_y_top = height - window_height * (window + 1)
		window_y_bottom = height - window_height * window
		# Draw the windows on the visualization image using cv2.rectangle()
		##TO DO
		cv2.rectangle(out_img, (window_x_left_min, window_y_top), (window_x_right_max, window_y_bottom),
					  (0, 255, 0), 2)
		####
		# Identify the nonzero pixels in x and y within the window
		##TO DO
		x_left_range = (nonzerox >= window_x_left_min) & (nonzerox <= window_x_left_max)
		x_right_range = (nonzerox >= window_x_right_min) & (nonzerox <= window_x_right_max)
		y_range = (nonzeroy >= window_y_top) & (nonzeroy <= window_y_bottom)
		window_nonzero_left = (x_left_range & y_range).nonzero()[0]
		window_nonzero_right = (x_right_range & y_range).nonzero()[0]
		####
		# Append these indices to the lists
		##TO DO
		left_lane_inds.append(window_nonzero_left)
		right_lane_inds.append(window_nonzero_right)

		####
		# If you found > minpix pixels, recenter next window on their mean position
		##TO DO
		if  len(window_nonzero_left) > minpix:
			leftx_current = np.mean(nonzerox[window_nonzero_left]).astype(np.int32)
		if len(window_nonzero_right) > minpix:
			rightx_current = np.mean(nonzerox[window_nonzero_right]).astype(np.int32)
		####

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each using np.polyfit()
	# If there isn't a good fit, meaning any of leftx, lefty, rightx, and righty are empty,
	# the second order polynomial is unable to be sovled.
	# Thus, it is unable to detect edges.
	try:
	##TODO
		left_fit = np.polyfit(lefty, leftx, deg=2)
		right_fit = np.polyfit(righty, rightx, deg=2)

		wps_left_y = np.linspace(0, max(lefty), 5).astype(int)
		wps_right_y = np.linspace(0, max(righty), 5).astype(int)

		x_left_poly = np.poly1d(left_fit)
		wps_left_x = x_left_poly(wps_left_y).astype(int)

		x_right_poly = np.poly1d(right_fit)
		wps_right_x = x_right_poly(wps_right_y).astype(int)

		# Interpolate if abnomality occurs
		shift = 400
		if wps_left_x[0] >= wps_right_x[0]:
			wps_left_x = wps_right_x - shift
			wps_right_x = wps_left_x + shift

		# Stack x, y to get points
		wps_left = np.stack((wps_left_x, wps_left_y), axis=1)
		wps_right = np.stack((wps_right_x, wps_right_y), axis=1)
		
		waypoints = (wps_left + wps_right)//2
		
		# Define sharp turn
		turn = "front"
		# When a sharp left turn is detected
		left_thresh = 1
		p1 = wps_right[0]
		p2 = wps_right[len(wps_right) - 1]
		slope = calculate_slope(p1, p2)
		
		if slope < left_thresh and slope > 0:
			wps_right_x_shifted = wps_right_x - shift
			waypoints = np.stack((wps_right_x_shifted, wps_right_y), axis=1)
			turn = "left"

		# When a sharp right turn is detected
		right_thresh = -1
		p1 = wps_left[0]
		p2 = wps_left[len(wps_left) - 1]
		slope = calculate_slope(p1, p2)
		
		if slope > right_thresh and slope < 0 :
			wps_left_x_shifted = wps_left_x + shift
			waypoints = np.stack((wps_left_x_shifted, wps_left_y), axis=1)
			turn = "right"
		
	####
	except TypeError:
		print("Unable to detect lanes")
		return None

	# Return a dict of relevant variables
	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['out_img'] = out_img
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds
	ret['waypoints'] = waypoints
	ret['wps_left'] = wps_left
	ret['wps_right'] = wps_right
	ret['turn'] = turn

	return ret


def tune_fit(binary_warped, left_fit, right_fit):
	"""
	Given a previously fit line, quickly try to find the line based on previous lines
	"""
	# Assume you now have a new warped binary image
	# from the next frame of video (also called "binary_warped")
	# It's now much easier to find line pixels!
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 0
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# If we don't find enough relevant points, return all None (this means error)
	min_inds = 10
	if lefty.shape[0] < min_inds or righty.shape[0] < min_inds:
		return None

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Return a dict of relevant variables
	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds

	return ret


def viz1(binary_warped, ret, save_file=None):
	"""
	Visualize each sliding window location and predicted lane lines, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	out_img = ret['out_img']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	if save_file is None:
		plt.show()
	else:
		plt.savefig(save_file)
	plt.gcf().clear()


def bird_fit(binary_warped, ret, mode="front", save_file=None):
	"""
	Visualize the predicted lane lines with margin, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Create an image to draw on and an image to show the selection window
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# if mode == "left":
	# 	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 0]
	# elif mode == "right":
	# 	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 0]

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	margin = 100  # NOTE: Keep this in sync with *_fit()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# if mode == "left":
	# 	# Draw the lane onto the warped blank image
	# 	cv2.fillPoly(window_img, np.int_([left_line_pts]), (255, 0, 255))
	# 	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	# 	plt.imshow(result)
	# 	plt.plot(left_fitx, ploty, color='yellow')
	# 	plt.xlim(0, 1280)
	# 	plt.ylim(720, 0)
	# elif mode == "right":
	# 	# Draw the lane onto the warped blank image
	# 	cv2.fillPoly(window_img, np.int_([right_line_pts]), (255, 0, 255))
	# 	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	# 	plt.imshow(result)
	# 	plt.plot(right_fitx, ploty, color='yellow')
	# 	plt.xlim(0, 1280)
	# 	plt.ylim(720, 0)
	# else:
	# 	# Draw the lane onto the warped blank image
	# 	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	# 	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	# 	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	# 	plt.imshow(result)
	# 	plt.plot(left_fitx, ploty, color='yellow')
	# 	plt.plot(right_fitx, ploty, color='yellow')
	# 	plt.xlim(0, 1280)
	# 	plt.ylim(720, 0)

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	plt.imshow(result)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)


	# cv2.imshow('bird',result)
	# cv2.imwrite('bird_from_cv2.png', result)

	# if save_file is None:
	# 	plt.show()
	# else:
	# 	plt.savefig(save_file)
	# plt.gcf().clear()

	return result


def final_viz(undist, m_inv, waypoints, wps_left, wps_right):
	"""
	Final lane line prediction visualized and overlayed on top of original image
	"""
	# Create an image to draw the lines on
	color_warp = np.zeros((720, 1280, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions

	# Draw the lane onto the warped blank image
	prev_c = waypoints[0]
	for c in waypoints:
		pc_x, pc_y = prev_c
		c_x, c_y = c
		cv2.circle(color_warp, (c_x, c_y), 20, (0, 0, 255), -1)
		cv2.line(color_warp, (pc_x, pc_y), (c_x, c_y), (255, 0, 0), 5)
		prev_c = c

	for c in wps_left:
		pc_x, pc_y = prev_c
		c_x, c_y = c
		cv2.circle(color_warp, (c_x, c_y), 20, (0, 255, 255), -1)

	for c in wps_right:
		pc_x, pc_y = prev_c
		c_x, c_y = c
		cv2.circle(color_warp, (c_x, c_y), 20, (0, 255, 0), -1)


	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
	# Combine the result with the original image
	# Convert arrays to 8 bit for later cv to ros image transfer
	undist = np.array(undist, dtype=np.uint8)
	newwarp = np.array(newwarp, dtype=np.uint8)
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	return result
