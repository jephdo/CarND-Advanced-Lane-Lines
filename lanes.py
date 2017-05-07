import glob
import collections

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# real life meters to pixels in the image for x and y directions
YM_PER_PIX = 30. / 720 
XM_PER_PIX = 3.7 / 700 


def is_close_parallel(left_fit, right_fit, relative_epsilon=2):
    """Check if the coefficientse of the estimated polynomials for left and 
    right lanes are similar enough."""
    A_left, A_right = left_fit[0], right_fit[0]
    B_left, B_right = left_fit[1], right_fit[1]
    A = abs(1.0 - A_left / A_right) < relative_epsilon
    B = abs(1.0 - B_left / B_right) < relative_epsilon
    return A and B


def eval_fit(fit, y):
    """Evaluate polynomial fit at a given point."""
    return fit[0] * y**2 + fit[1] * y + fit[2]


def compute_curvature(fit):
    """Measure radius of curvature given polynomial fit. Returns curvature
    in meters."""
    ys = np.array(np.linspace(0, 719, num=10))
    xs = np.array([eval_fit(fit, y) for y in ys])
    y_eval = np.max(y)

    fit_cr = np.polyfit(ys * YM_PER_PIX, xs * XM_PER_PIX, 2)
    curverad = ((1 + (2 * fit_cr[0] * y_eval / 2. + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

    return curverad


def compute_offset(left_fit, right_fit, image_size):
    """Given polynomial fits for left and right lanes, determine how far offset
    the car is from the center point of the lane in meters."""
    length, width = image_size
    left_lane = eval_fit(left_fit, length) 
    right_lane = eval_fit(right_fit, length)
    center = (left_lane + right_lane) / 2.
    offset = (center - width / 2.) * XM_PER_PIX
    return offset


def threshold_by_color_and_gradient(img, gray_thresh=(40, 100), 
    s_thresh=(170, 255), l_thresh=(30, 255), kernel_size=3):
    img = np.copy(img)

    # convert to HLS color space.
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    l_channel = hls[:, :, 1]
    
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobelx = np.absolute(sobelx)
    sobelx = np.uint8(255 * sobelx / np.max(sobelx))

    thresh_min, thresh_max = gray_thresh
    sx_binary = np.zeros_like(sobelx)
    sx_binary[(sobelx >= thresh_min) & (sobelx <= thresh_max)] = 1

    thresh_min, thresh_max = s_thresh
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh_min) & (s_channel <= thresh_max)] = 1

    thresh_min, thresh_max = l_thresh
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= thresh_min) & (l_channel <= thresh_max)] = 1

    # combine binary image
    combined = np.zeros_like(sx_binary)
    combined[((l_binary == 1) & (s_binary == 1) | (sx_binary == 1))] = 1
#     combined[(s_binary == 1) | (sx_binary == 1)] = 1
    return combined


class Camera(object):
    
    pattern_size = (9, 6)
    
    def __init__(self, calibration_images='./camera_cal/calibration*.jpg'):
        self.is_calibrated = False
        self.calibration_images = calibration_images
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        
    def calibrate(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(self.calibration_images)

        # Step through the list and search for chessboard corners
        for fname in images:
            img = mpimg.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
        assert objpoints and imgpoints, 'No object or image points found to calibrate with'
        
        image_size = gray.shape[::-1]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, 
                                                           image_size, None, None)
        
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.is_calibrated = True
    
    def undistort(self, img):
        if not self.is_calibrated:
            raise ValueError("Camera not yet calibrated with chessboard images.")
        
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)


class Perspective(object):

    # these points were determined manually by looking at images and thru
    # trial and error
    src = np.float32([
        [  253.,   697.],
        [  585.,   456.],
        [  700.,   456.],
        [ 1061.,   690.]
    ])

    dst = np.float32([
        [  303.,   697.],
        [  303.,     0.],
        [ 1011.,     0.],
        [ 1011.,   690.]
    ])
    
    def __init__(self, src=None, dst=None):
        self.src = src or self.src
        self.dst = dst or self.dst
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inverse = cv2.getPerspectiveTransform(self.dst, self.src)
    
    def warp(self, img):
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), 
                                   flags=cv2.INTER_LINEAR)
    
    def unwarp(self, img):
        return cv2.warpPerspective(img, self.M_inverse, 
                                   (img.shape[1], img.shape[0]), 
                                   flags=cv2.INTER_LINEAR)


class Lines(object):

    ROLLING_WINDOW = 4
    MAX_SKIPS = 30

    def __init__(self):
        self.camera = Camera()
        self.camera.calibrate()
        self.perspective = Perspective()

        # debugging stats 
        self.skip_counter = 0
        self.skipped_frames = {}
        self.frame_counter = 0

        # I use a deque data structure to maintain a rolling list of 
        # lane fits from previous frames
        self.prior_fits = collections.deque(maxlen=self.ROLLING_WINDOW)

    def process_image(self, img):
        """Processes a raw image into a binary warped image."""
        img = self.camera.undistort(img)
        binary = threshold_by_color_and_gradient(img)
        binary_warped = self.perspective.warp(binary)
        return binary_warped

    def is_fit_sane(self, left_fit, right_fit):
        """Check if the left and right fits have expected parameters."""
        if not is_close_parallel(left_fit, right_fit):
            return False

        left_curve = compute_curvature(left_fit)
        right_curve = compute_curvature(right_fit)

        if (1.0 - left_curve / right_curve) > 0.6:
            return False

        return True

    def fit_lines(self, binary_warped):
        """I use the sliding window approach directly from the lecture."""
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        nwindows = 9
        
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
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
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
    #         cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 4) 
    #         cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 4) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit


    def determine_fit_to_draw(self, img, warped):
        """This function determines the actual polynomials to use for drawing
        lane lines. Every image gets fit with polynomials, however, if the
        fit is bad then those are skipped. Additionally, this returns a "rolling
        average" of the past N successful frames."""
        left_fit, right_fit = self.fit_lines(warped)

        if self.is_fit_sane(left_fit, right_fit):
            self.skip_counter = 0
            self.prior_fits.append((left_fit, right_fit))
        else:
            # print('skipped', self.frame_counter)
            # if self.skip_counter > self.MAX_SKIPS:
                # raise RuntimeError("Skipped too many frames -- not able to generate a good fit for lane lines.")
            self.skip_counter += 1
            self.skipped_frames[self.frame_counter] = {'img': img, 'warped': warped}

        left_fits, right_fits = zip(*self.prior_fits)
        left_fit = np.mean(left_fits, axis=0)
        right_fit = np.mean(right_fits, axis=0)
        return left_fit, right_fit


    def draw(self, img):
        """Draws lane lines onto the given image. This is the main method called
        for creating the video."""
        warped = self.process_image(img)
        left_fit, right_fit = self.determine_fit_to_draw(img, warped)

        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        newwarp = self.perspective.unwarp(color_warp)
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        self.frame_counter += 1

        result_with_text = self.draw_text(result, left_fit, right_fit)
        return result_with_text

    def draw_text(self, img, left_fit, right_fit):
        """Draws information about the lane offset and curvatures onto the 
        image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        left_curvature = compute_curvature(left_fit)
        right_curvature = compute_curvature(right_fit)
        curvature_text = 'Left Curvature: %d(m) | Right Curvature: %d(m)' % (left_curvature, right_curvature)
        cv2.putText(img, curvature_text, (100, 50), font, fontScale=1, 
                    color=(255, 255, 255), thickness=1)
        
        offset = compute_offset(left_fit, right_fit, img.shape[:2])
        offset_text = 'Vehicle is %.2fm off of center' % offset
        cv2.putText(img, offset_text, (100, 90), font, fontScale=1, 
                    color=(255, 255, 255), thickness=1)
        return img



def create_video(input_file='./project_video.mp4', output_file='./lane_lines.mp4'):
    from moviepy.editor import VideoFileClip

    lanes = Lines()
    clip = VideoFileClip(input_file)
    out_clip = clip.fl_image(lanes.draw) 
    out_clip.write_videofile(output_file, audio=False)

    return lanes