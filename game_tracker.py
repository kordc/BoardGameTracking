import cv2
import numpy as np
import utils

class CycladesTracker:
    def __init__(self,):
        self.blur = False
        self.equalize = "local"
        self.foreground_knn = cv2.createBackgroundSubtractorKNN()

    def find_separating_line(self, frame):
        # Find point dividing left and right part of the board
        g = np.ones((10,10))/100
        g2 = -np.ones((10,10))/100


        fg_cv = cv2.filter2D(frame[:,:,0], -1, g)
        fg_cv2 = cv2.filter2D(frame[:,:,2].astype(g2.dtype), -1, g2)

        filtered = (np.maximum(np.zeros_like(fg_cv), fg_cv + fg_cv2) > 20).sum(axis=0)

        line_x = np.where(filtered > 300)[0][0]

        return line_x - 10

    def separate(self, frame_c, frame_gray):
        # separates everything
        return frame_c[:,:self.intersecting_line_x], frame_gray[:,:self.intersecting_line_x],\
                     frame_c[:,self.intersecting_line_x:], frame_gray[:,self.intersecting_line_x:]
    
    def preprocess_each_frame(self,frame):
        # make processing that is applied to every frame
        frame = cv2.resize(frame, None,fx=0.4, fy=0.4)
        if self.blur:
            frame = cv2.GaussianBlur(frame, (3,3), 0)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.equalize == "global":
            frame_gray = cv2.equalizeHist(frame_gray)
        elif self.equalize == "local":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            frame_gray = clahe.apply(frame_gray)

        return frame, frame_gray

    def label_circles(self, circles, frame):
        # label circles with sea or land 
        labeled_circles = []
        for circle in circles[0,:]:
            x,y,r = circle
            
            x1,y1 = x - r, y - r # making square out of a circle
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

    def update_interesting_objects(self, foreground, frame):
        # bases on foreground it updates interesting objects
        cnts, hier = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            boxes.append([x,y,w,h]) # Getting coordinates of every new found box
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        return frame

        # if not candidates: # if this is first iteration
        #     candidates = {tuple(box): 1 for box in boxes}
        # else:
        #     old = list(candidates.keys()) #previous candidates
        #     new = boxes # currently found boxes

        #     matches = centres_within(np.array(old), np.array(new)) # which box match to whom
        #     new_candidates = {tuple(box): 1 for box in boxes}
        #     for match in matches: # Box ith from previous iteration matched to jth from this iteration
        #         i,j = match
        #         new_candidates[tuple(new[i])] = candidates[old[j]] + 1
        #         if new_candidates[tuple(new[i])] == 3: # If box was seen twice in the same place we track it
        #             correct_boxes.append(new[i])
        #             x,y,w,h = new[i]
        #             cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #     candidates = new_candidates


    def run(self, video_path):
        # At first processing of the first frame
        video, width, height, fps = utils.get_video(video_path)

        first_frame = utils.get_one_frame(video, frame_num=0, current_frame=0)
        frame_color, frame_gray = self.preprocess_each_frame(first_frame)

        self.intersecting_line_x = self.find_separating_line(frame_color)


        self.left_part_color, self.left_part_gray, self.right_part_color, self.right_part_gray = self.separate(frame_color, frame_gray)

        self.map_circles = utils.find_circles(self.right_part_gray, equalize=None, minDist=30, param1=170, param2=20, minRadius=12, maxRadius=25)
        self.map_circles = self.label_circles(self.map_circles, self.right_part_color)


        current_frame = 0

        #than processing of later ones withouth unnecessary steps, just updates
        while video.isOpened():
            #video.set(cv2.CAP_PROP_POS_FRAMES, current_frame*10) # 3fps 
            ret, frame = video.read()
            
            if ret:
                current_frame +=1
                frame_color, frame_gray = self.preprocess_each_frame(frame)
                
                self.left_part_color, self.left_part_gray, self.right_part_color, self.right_part_gray = self.separate(frame_color,frame_gray)

                foreground = self.foreground_knn.apply(cv2.GaussianBlur(frame_color, (3,3), 0))
                foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((7,7), dtype=np.uint8))
                frame_color = self.update_interesting_objects(foreground, frame_color)

                #self.right_part = self.draw_circles(self.right_part_color, self.map_circles)
                cv2.imshow("left", self.left_part_color)
                cv2.imshow("right", self.right_part_gray)
                cv2.imshow("game look", np.concatenate([self.left_part_color, self.right_part_color], axis=1))
                cv2.imshow("foreground", foreground)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
           

if __name__ == "__main__":
    tracker = CycladesTracker()
    tracker.run("data/cyklady_lvl1_1.mp4")
