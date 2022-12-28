import cv2
import numpy as np
import utils

class CycladesTracker:
    def __init__(self, empty_board_path):
        self.blur = False
        self.equalize = "local"
        
        self.empty_board_image = cv2.imread(empty_board_path)

        self.MAX_FEATURES = 500
        self.GOOD_MATCH_PERCENT = 0.15
        self.orb = cv2.ORB_create(self.MAX_FEATURES)

        self.objects = {'red_ships': [], 'yellow_ships': [], 'black_ships': [],
                        'red_counters': [], 'yellow_counters': [], 'black_counters': [],
                        'cards': []}

    def find_separating_line(self, frame):
        # Find point dividing left and right part of the board
        g = np.ones((10,10))/100
        g2 = -np.ones((10,10))/100


        fg_cv = cv2.filter2D(frame[:,:,0], -1, g)
        fg_cv2 = cv2.filter2D(frame[:,:,2].astype(g2.dtype), -1, g2)

        filtered = (np.maximum(np.zeros_like(fg_cv), fg_cv + fg_cv2) > 20).sum(axis=0)

        line_x = np.where(filtered > 300)[0][0]

        return line_x - 10

    def equalize_color_image(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab_planes = np.array(cv2.split(lab))
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(3,3))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return bgr

    def separate(self, frame_c, frame_gray):
        # separates everything
        return frame_c[:,:self.intersecting_line_x], frame_gray[:,:self.intersecting_line_x],\
                     frame_c[:,self.intersecting_line_x:], frame_gray[:,self.intersecting_line_x:]
    
    def preprocess_each_frame(self,frame):
        # make processing that is applied to every frame
        frame = cv2.resize(frame, None,fx=0.4, fy=0.4)
        if self.blur:
            frame = cv2.GaussianBlur(frame, (3,3), 0)
        
        if self.equalize: # equalize color image!
            frame = self.equalize_color_image(frame)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # if self.equalize == "global":
        #     frame_gray = cv2.equalizeHist(frame_gray)
        # elif self.equalize == "local":
        #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #     frame_gray = clahe.apply(frame_gray)

        return frame, frame_gray

    def label_circles(self, circles, frame):
        # label circles with sea or land 
        labeled_circles = []
        for circle in circles[0,:]:
            x,y,r = circle
            
            x1,y1 = x - min([x,r]), y - min([y,r]) # making square out of a circle
            x2,y2 = x + r, y + r

            circle[2] = 19 # Setting constant radius to the circle

            place = frame[y1:y2, x1:x2]

            sea_mask = utils.segment_by_hsv_color(place,np.array([50,50,50]),np.array([110,255,255]))
            land_mask = utils.segment_by_hsv_color(place,np.array([10,50,50]),np.array([49,255,255]))

            total = (x2-x1) * (y2-y1) * 255
            sea_ratio = np.sum(sea_mask) / total
            land_ratio = np.sum(land_mask)/ total

            if land_ratio > 0.08:
                labeled_circles.append((circle, "land"))
            elif sea_ratio > 0.7:
                labeled_circles.append((circle, "sea"))
            
        return labeled_circles

    def draw_circles(self, frame, labeled_circles):
        # draw labeled circles
        for lab_circle in labeled_circles:
            circle, label = lab_circle
            if label == "sea":
                cv2.circle(frame, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)
            elif label == "land":
                cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            else:
                cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)

        return frame

    def classify_objects(self, x, y, w, h, frame):
        yellow = utils.segment_by_hsv_color(frame[y:y+h, x:x+w], np.array([20, 100, 100]), np.array([30, 255, 255]))
        black = utils.segment_by_hsv_color(frame[y:y+h, x:x+w], np.array([0, 0, 0]), np.array([180, 255, 30]))
        red = utils.segment_by_hsv_color(frame[y:y+h, x:x+w], np.array([0, 100, 100]), np.array([10, 255, 255]))
        sea_mask = utils.segment_by_hsv_color(
            frame[y:y+h, x:x+w], np.array([50, 50, 50]), np.array([110, 255, 255]))
        land_mask = utils.segment_by_hsv_color(
            frame[y:y+h, x:x+w], np.array([10, 50, 50]), np.array([49, 255, 255]))

        total = w * h * 255
        sea_ratio = np.sum(sea_mask) / total
        land_ratio = np.sum(land_mask) / total

        obj_type = ""
        # if land_ratio > 0.01:
        #     obj_type = "land"
        # elif sea_ratio > 0.2:
        #     obj_type = "ship"
        # else:
        #     obj_type = "cardboard"

        if black.sum() > black.size * 0.8 and black.sum() > red.sum() and black.sum() > yellow.sum():
            cv2.putText(frame, "black " + obj_type, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.objects['black_ships'].append([x, y, w, h])
            return True
        if red.sum() > red.size * 0.8 and red.sum() > yellow.sum() and red.sum() > black.sum():
            cv2.putText(frame, "red " + obj_type, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.objects['red_ships'].append([x, y, w, h])
            return True
        if yellow.sum() > yellow.size * 0.8 and yellow.sum() > red.sum() and yellow.sum() > black.sum():
            cv2.putText(frame, "yellow " + obj_type, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.objects['yellow_ships'].append([x, y, w, h])
            return True

        return False
    
    def update_interesting_objects(self, foreground, frame, candidates):
        # bases on foreground it updates interesting objects
        cnts, hier = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes, correct_boxes = [], []
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            boxes.append([x,y,w,h]) # Getting coordinates of every new found box
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            # _ = self.classify_objects(x, y, w, h, frame)
        
        if not candidates: # if this is first iteration
            candidates = {tuple(box): 1 for box in boxes}
        else:
            old = list(candidates.keys()) #previous candidates
            new = boxes # currently found boxes

            matches = utils.centres_within(np.array(old), np.array(new)) # which box match to whom
            new_candidates = {tuple(box): 1 for box in boxes}
            for match in matches: # Box ith from previous iteration matched to jth from this iteration
                i,j = match
                new_candidates[tuple(new[i])] = candidates[old[j]] + 1
                if new_candidates[tuple(new[i])] == 3: # If box was seen twice in the same place we track it
                    correct_boxes.append(new[i])
                    x,y,w,h = new[i]
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    _ = self.classify_objects(x, y, w, h, frame)
            candidates = new_candidates
        return frame, candidates


    def alignImageToFirstFrame(self, im_gray, im_color):
        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(self.MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im_gray, None)
        
        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = list(matcher.match(descriptors1, self.first_frame_desc, None))
        
        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)
        
        # Remove not so good matches
        numGoodMatches = int(len(matches) * self.GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]
        
        # Draw top matches
        imMatches = cv2.drawMatches(im_color, keypoints1, self.first_frame_color, self.first_frame_key, matches, None)
        cv2.imwrite("matches.jpg", imMatches)
        
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

    def initialize_background_subtractor(self, images = None):
        foreground_knn = cv2.createBackgroundSubtractorKNN()

        if images is None: #! First every initialization
            empty_board_color, empty_board_gray = self.preprocess_each_frame(self.empty_board_image)
            empty_board_color = self.alignImageToFirstFrame(empty_board_gray, empty_board_color)
        else: #! reinitialization
            empty_board_color, empty_board_gray = images

        for i in range(20):
            foreground_knn.apply(empty_board_color)

        return foreground_knn

    def analyze_left_part(left_color, left_gray):
        cv2.imshow("left_color", left_color)
        cv2.imshow("left_gray", left_gray)

    def initialize_first_frame(self, first_frame, first_frame_gray=None):
        self.first_frame_color, self.first_frame_gray = self.preprocess_each_frame(first_frame)
        self.height, self.width = self.first_frame_gray.shape

        self.first_frame_key, self.first_frame_desc = self.orb.detectAndCompute(self.first_frame_gray, None)
 
        self.intersecting_line_x = self.find_separating_line(self.first_frame_color)

        self.left_part_color, self.left_part_gray, self.right_part_color, self.right_part_gray = self.separate(self.first_frame_color, self.first_frame_gray)

        self.map_circles = utils.find_circles(self.right_part_gray, equalize=None, minDist=30, param1=170, param2=20, minRadius=12, maxRadius=25)
        self.map_circles = self.label_circles(self.map_circles, self.right_part_color)
        self.foreground_knn = self.initialize_background_subtractor()

    def reinitialize_first_frame(self, frame_color, frame_gray):
        self.first_frame_color, self.first_frame_gray = frame_color, frame_gray

        self.first_frame_key, self.first_frame_desc = self.orb.detectAndCompute(self.first_frame_gray, None)

        #self.foreground_knn = self.initialize_background_subtractor((frame_color, frame_gray))

    def run(self, video_path):
        # At first processing of the first frame
        video, width, height, fps = utils.get_video(video_path)

        first_frame = utils.get_one_frame(video, frame_num=0, current_frame=0)

        self.initialize_first_frame(first_frame)

        current_frame = 0
        candidates = None

        #than processing of later ones withouth unnecessary steps, just updates
        while video.isOpened():
            video.set(cv2.CAP_PROP_POS_FRAMES, current_frame) 
            ret, frame = video.read()
            
            if ret:
                current_frame += 10 # 3fps
                frame_color, frame_gray = self.preprocess_each_frame(frame)
                frame_color = self.alignImageToFirstFrame(frame_gray, frame_color) #! I don't know where exactly this should be done so that grayscale images is warped too... to be discussed

                

                self.left_part_color, self.left_part_gray, self.right_part_color, self.right_part_gray = self.separate(frame_color,frame_gray)

                foreground = self.foreground_knn.apply(cv2.GaussianBlur(frame_color, (3,3), 0))
                foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((7,7), dtype=np.uint8))
                
                frame_color, candidates = self.update_interesting_objects(
                    foreground, frame_color, candidates)

                #! WE reupdate our first frame if we detect some problems with foreground
                #! This actually may make things even worse if we do this in wrong moment e.g when we have hand on the screen it became background
                # if current_frame % 300 == 0:
                #     print("REINITIALIZE")
                #     self.reinitialize_first_frame(frame_color, cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)) #! gray frame is not warped by default this should be rethought


                self.right_part = self.draw_circles(self.right_part_color, self.map_circles)

                #self.analyze_left_part(self.left_part_color, self.left_part_gray)

                h = 0
                self.stats = np.zeros(
                    self.right_part_color.shape, dtype=np.uint8)
                self.stats.fill(255)
                for key, l in self.objects.items():
                    cv2.putText(self.stats, key + ": " + str(len(l)), (20,
                                20 + 20 * h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    h += 1
                    
                cv2.imshow("left", self.left_part_color)
                cv2.imshow("right", self.right_part_color)
                cv2.imshow("game look", np.concatenate([self.left_part_color, self.right_part_color], axis=1))
                cv2.imshow("foreground", foreground)
  
                cv2.imshow("stats", self.stats)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
           

if __name__ == "__main__":
    tracker = CycladesTracker(empty_board_path="empty_board.jpg")
    tracker.run("data/cyklady_lvl1_1.mp4")
