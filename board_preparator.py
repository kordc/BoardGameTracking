import cv2
import numpy as np
import utils

class BoardPreparator():
    def __init__(self, empty_board_path):
        self.MAX_FEATURES = 500
        self.GOOD_MATCH_PERCENT = 0.15
        self.orb = cv2.ORB_create(self.MAX_FEATURES)

        self.blur = False
        self.equalize = "local"

        self.empty_board_image = cv2.imread(empty_board_path)
        self.empty_board_color, self.empty_board_gray = self.preprocess_each_frame(
            self.empty_board_image)

    def find_separating_line(self, frame):
        # Find point dividing left and right part of the board
        g = np.ones((10, 10))/100
        g2 = -np.ones((10, 10))/100

        fg_cv = cv2.filter2D(frame[:, :, 0], -1, g)
        fg_cv2 = cv2.filter2D(frame[:, :, 2].astype(g2.dtype), -1, g2)

        fg_cv = cv2.filter2D(frame[:,:,0], -1, g)
        fg_cv2 = cv2.filter2D(frame[:,:,2].astype(g2.dtype), -1, g2)

        filtered = (np.maximum(np.zeros_like(fg_cv), fg_cv + fg_cv2) > 20).sum(axis=0)
        line_x = np.where(filtered > 200)[0][0]

        return line_x - 10
    
    def equalize_color_image(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab_planes = np.array(cv2.split(lab))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return bgr

    def separate(self, frame):
        # separates everything into two halves
        return frame[:, :self.intersecting_line_x], frame[:, self.intersecting_line_x:]

    def preprocess_each_frame(self, frame):
        # make processing that is applied to every frame
        frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
        if self.blur:
            frame = cv2.GaussianBlur(frame, (3, 3), 0)

        if self.equalize:  # equalize color image!
            frame = self.equalize_color_image(frame)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame, frame_gray

    def alignImageToFirstFrame(self, im_gray, im_color):
        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(self.MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im_gray, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = list(matcher.match(
            descriptors1, self.first_frame_desc, None))

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * self.GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        #imMatches = cv2.drawMatches(im_color, keypoints1, self.first_frame_color, self.first_frame_key, matches, None)
        #cv2.imwrite("matches.jpg", imMatches)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = self.first_frame_key[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        height, width, channels = self.first_frame_color.shape
        im1Reg = cv2.warpPerspective(im_color, h, (width, height))

        return im1Reg

    def initialize_background_subtractor(self, images=None):
        foreground_knn = cv2.createBackgroundSubtractorKNN()

        if images is None:  # ! First every initialization
            empty_board_color = self.alignImageToFirstFrame(
                self.empty_board_gray, self.empty_board_color)
        else:  # ! reinitialization, but not used
            empty_board_color, empty_board_gray = images

        for i in range(10):
            foreground_knn.apply(empty_board_color)

        return foreground_knn

    def initialize_first_frame(self, first_frame):
        self.first_frame_color, self.first_frame_gray = self.preprocess_each_frame(
            first_frame)
        self.height, self.width = self.first_frame_gray.shape

        self.first_frame_key, self.first_frame_desc = self.orb.detectAndCompute(
            self.first_frame_gray, None)

        self.intersecting_line_x = self.find_separating_line(self.first_frame_color)

        self.foreground_knn = self.initialize_background_subtractor()

        left_part_color, right_part_color = self.separate(
            self.first_frame_color)
        left_part_gray, right_part_gray = self.separate(
            self.first_frame_gray)


        return right_part_color, right_part_gray

    def reinitialize_first_frame(self, frame_color, frame_gray):
        # reinitializes keypoints for alignment
        self.first_frame_color, self.first_frame_gray = frame_color, frame_gray

        self.first_frame_key, self.first_frame_desc = self.orb.detectAndCompute(
            self.first_frame_gray, None)

        #! we thought about subtractor reinitialization but it worked bad 
        #self.foreground_knn = self.initialize_background_subtractor((frame_color, frame_gray))

    def get_mask_of_left_mess(self):
        # Cut of the left background
        edges = cv2.Canny(cv2.medianBlur(
            self.empty_board_color, 3), 200, 250, apertureSize=3)
        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, None, 400, 20)
        if linesP is not None:
            line = linesP[0][0]
        else: #! We have very strange problem that we used the same image and same opencv version and on one PC it was working and on other not.
            line = np.array([14, 424, 100, 14])
        bounding_points = np.linspace(
            line[2]-5, line[3]-5, self.empty_board_color.shape[0]).astype(np.uint8)

        mask = np.ones_like(self.empty_board_gray)
        for i, boundary in enumerate(bounding_points):
            mask[i, :boundary] = 0

        self.mask = mask

    def get_foreground(self, frame_color):
        foreground = self.foreground_knn.apply(cv2.GaussianBlur(frame_color, (3, 3), 0))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones(
            (7, 7), dtype=np.uint8)) * self.mask  # To filter defined background we multiply by mask

        # To have only very intense foreground
        foreground = ((foreground > 200) * 255).astype(np.uint8)
        return self.separate(foreground)

    def process(self, frame, current_frame):
        frame_color, frame_gray = self.preprocess_each_frame(frame)

        frame_color = self.alignImageToFirstFrame(frame_gray, frame_color)

        left_part_color, right_part_color = self.separate(frame_color)

        left_foreground, right_foreground = self.get_foreground(frame_color)
        
        if current_frame % 300 == 0:
            self.reinitialize_first_frame(frame_color, cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY))

        return left_part_color, left_foreground, right_part_color, right_foreground

    def initialize(self, first_frame):
        self.get_mask_of_left_mess()
        return self.initialize_first_frame(first_frame)