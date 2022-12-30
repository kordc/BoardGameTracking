import cv2
import numpy as np
import utils

from board_preparator import BoardPreparator
from left_part_analyzer import LeftPartAnalyzer
class CycladesTracker:
    def __init__(self, board_preparator: BoardPreparator, left_part_analyzer: LeftPartAnalyzer):
        self.board_preparator = board_preparator
        self.left_part_analyzer = left_part_analyzer

        self.objects = {}
        
        self.moved = False
        self.placed = False
        self.waiting_moved = None
        self.waiting_placed = None 


    def label_circles(self, circles, frame):
        # label circles with sea or land
        self.segmented_right_part = frame.copy()
        #self.segmented_right_part[:,:,0] = 255

        labeled_circles = []
        for circle in circles[0, :]:
            x, y, r = circle

            # making square out of a circle
            x1, y1 = max(int(x) - r, 0), max(int(y) - r, 0)
            x2, y2 = x + r, y + r

            circle[2] = 19  # Setting constant radius to the circle

            place = frame[y1:y2, x1:x2]

            sea_mask = utils.get_sea_mask(place, label_circles=True)
            land_mask = utils.get_land_mask(place, label_circles=True)

            total = utils.calculate_total(x2-x1, y2-y1)
            sea_ratio = np.sum(sea_mask) / total
            land_ratio = np.sum(land_mask) / total

            if land_ratio > 0.08:
                labeled_circles.append((circle, "land"))
                self.segmented_right_part[y1:y2, x1:x2] = (0, 255, 0)
            elif sea_ratio > 0.7:
                labeled_circles.append((circle, "sea"))
                self.segmented_right_part[y1:y2, x1:x2] = (255, 0, 0)

        return labeled_circles

    def draw_circles(self, frame, labeled_circles):
        # draw labeled circles
        for lab_circle in labeled_circles:
            circle, label = lab_circle
            if label == "sea":
                cv2.circle(frame, (circle[0], circle[1]),
                           circle[2], (255, 0, 0), 2)
            elif label == "land":
                cv2.circle(frame, (circle[0], circle[1]),
                           circle[2], (0, 255, 0), 2)
            else:
                cv2.circle(frame, (circle[0], circle[1]),
                           circle[2], (0, 0, 255), 2)

        return frame

    def detect_islands(self, frame):
        land_mask = utils.get_land_mask(self.segmented_right_part)
        land_mask = cv2.erode(land_mask, np.ones((3, 3)), iterations=1)
        land_mask = cv2.dilate(land_mask, np.ones((3, 3)), iterations=6)
        cnts, hier = cv2.findContours(
            land_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        islands = {}
        for cnt in cnts:
            if cv2.contourArea(cnt) > 10000:
                continue
            if cnt.shape[0] > 5:
                xr, yr, wr, hr = cv2.boundingRect(cnt)
                islands[(xr, yr, wr, hr)] = [cnt]
        return islands

    def detect_current_island(self, x, y, w, h, frame):
        for island in self.islands.keys():
            xr, yr, wr, hr = island
            if x > xr*0.8 and x+w < (xr+wr)*1.2 and y > yr*0.8 and y+h < (yr+hr)*1.2:
                return self.islands[island]
        return None

    def object_type(self, x, y, w, h, frame):
        sea_mask = utils.get_sea_mask(self.segmented_right_part[y:y+h, x:x+w])
        land_mask = utils.get_land_mask(self.segmented_right_part[y:y+h, x:x+w])
    
        total = utils.calculate_total(w,h)
        sea_ratio = np.sum(sea_mask) / total
        land_ratio = np.sum(land_mask) / total

        obj_type = "unknown"
        if land_ratio > sea_ratio and land_ratio > 0.1:
            obj_type = "warrior"
        elif sea_ratio > land_ratio and sea_ratio > 0.1:
            obj_type = "ship"

        return obj_type

    def object_color(self, yellow, black, red):
        if yellow.sum() > yellow.size * 0.8 and yellow.sum() > red.sum() and yellow.sum() > black.sum():
            return "yellow"
        elif black.sum() > black.size * 0.8 and black.sum() > red.sum():
            return "black"
        elif red.sum() > red.size * 0.8:
            return "red"
        else:
            return "unknown"

    def get_font_color(self, color):
        if color == "yellow":
            font_color = (0, 255, 255)
        elif color == "black":
            font_color = (0, 0, 0)
        elif color == "red":
            font_color = (0, 0, 255)
        else:
            font_color = (255, 255, 255)
        return font_color

    def is_moved(self, name):
        color = name.split(" ")[0]
        if name not in self.objects.keys(): 
            return -1

        for i, obj_box in enumerate(self.objects[name]):
            cutted= utils.cut_obj(self.right_part_color, obj_box)
            yellow, black, red = utils.segment_colors(cutted)
            new_color = self.object_color(yellow, black, red)
            if new_color != color:
                return i

        return -1

    def classify_right_objects(self, box, frame):
        x, y, w, h = box
        yellow, black, red = utils.segment_colors(self.right_part_color[y:y+h, x:x+w])

        color = self.object_color(yellow, black, red)
        obj_type = self.object_type(x, y, w, h, frame)

        name = color + " " + obj_type

        if "unknown" in name or w*h > 1000 or w*h < 150: #! Filter too big or too small objects
            return

        if name in self.objects.keys():
            for coords in self.objects[name]:
                one = abs(coords[0] - x) < 10
                two = abs(coords[1] - y) < 10
                three = abs(coords[2] - w) < 10
                four = abs(coords[3] - h) < 10
                if one and two and three and four:
                    return

        
        if obj_type == "warrior":
            current_island = self.detect_current_island(x, y, w, h, frame)
            if current_island is not None:
                ellipse = cv2.fitEllipse(current_island[0])
                xr, yr, wr, hr = cv2.boundingRect(current_island[0])
                new_owner = False
                if len(self.islands[(xr, yr, wr, hr)]) > 1:
                    if self.islands[(xr, yr, wr, hr)][1] != color:
                        new_owner = True
                self.islands[(xr, yr, wr, hr)] = [
                    current_island[0], color, ellipse, new_owner]

        if self.is_moved(name) != -1:
            self.objects[name].pop(self.is_moved(name))
            self.moved = True

        if not name in self.objects.keys():
            self.objects[name] = [[x, y, w, h]]
            self.placed = True
        else:
            self.objects[name].append([x, y, w, h])
            self.placed = True

    def update_interesting_objects(self, foreground, frame, candidates, left=False, debug_contours = False):
        # bases on foreground it updates interesting objects
        cnts, hier = cv2.findContours(
            foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes, correct_boxes = [], [] # boxes found and boxes containing actual object
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)

            #neglect very small boxes
            if w*h > 80 and w*h < 12000: 
                boxes.append([x,y,w,h]) # Getting coordinates of every new found box
                
                #For debugging purposes
                if debug_contours:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
       
        #there is no valid box to be processed further
        if len(boxes) == 0:
            return frame, candidates

        if not candidates: # if this is first iteration
            candidates = {tuple(box): 1 for box in boxes}
        else:
            old = list(candidates.keys()) #previous candidates
            new = boxes # currently found boxes
            matches = utils.centres_within(np.array(old), np.array(new)) # which old box match to which new box
            new_candidates = {tuple(box): 1 for box in boxes}
            for match in matches: # Box ith from previous iteration matched to jth from this iteration
                i,j = match
                #If there is a match we increase counter of old candidate by 1
                new_candidates[tuple(new[i])] = candidates[old[j]] + 1
                #! This equality here may be problematic as sometimes more than one box can be matched potentially!
                # If box was seen for few times we check if it contains an object
                if new_candidates[tuple(new[i])] == 3:
                    if debug_contours:
                        cv2.rectangle(frame, (x, y), (x+w, y+h),(255, 0, 0), 2)

                    correct_boxes.append(new[i])
                    box = new[i]
                    
                  
                    self.classify_right_objects(box, frame)
                        
            candidates = new_candidates
        return frame, candidates

    def update_view(self):
        h = 0
        self.stats = np.zeros(
            self.right_part_color.shape, dtype=np.uint8)
        self.stats.fill(255)
        for key, l in self.objects.items():
            cv2.putText(self.stats, key + ": " + str(len(l)), (20,
                        20 + 20 * h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            h += 1

        taken_islands = sum(
            [1 for island, stats in self.islands.items() if len(stats) > 1])
        color_islands = {'red': 0, 'black': 0, 'yellow': 0}
        for island in self.islands.values():
            if len(island) > 1:
                color_islands[island[1]] += 1
        for color, cnt in color_islands.items():
            cv2.putText(self.stats, color + " islands: " + str(cnt), (20,
                                                                      20 + 20 * h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            h += 1
        cv2.putText(self.stats, "taken islands: " + str(taken_islands), (20,
                                                                         20 + 20 * h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        h += 1
        if self.moved and self.waiting_moved is None:
            self.waiting_moved = 9
        elif self.moved:
            self.waiting_moved -= 1
            if self.waiting_moved == 0:
                self.moved = False
                self.waiting_moved = None
        cv2.putText(self.stats, "moved_counter: " + str(self.moved), (20,
                                                                   20 + 20 * h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        h += 1

        for island, stats in self.islands.items():
            if len(stats) <= 1:
                continue
            cnt, color, ellipse, new_owner = stats
            xr, yr, wr, hr = island
            text = "island"
            if new_owner is not None:
                text += " " + str(new_owner)
            cv2.ellipse(self.right_part_color, ellipse, (0, 255, 0), 2)
            cv2.putText(self.right_part_color, "island", (xr, yr),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, self.get_font_color(color), 2)

        for object in self.objects.keys():
            for x, y, w, h in self.objects[object]:
                color, obj_type = object.split(" ")
                cv2.rectangle(self.right_part_color, (x, y),
                              (x + w, y + h), self.get_font_color(color), 2)
                cv2.putText(self.right_part_color, obj_type, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, self.get_font_color(color), 2)
        

    def run(self, video_path):
        # At first processing of the first frame
        video, width, height, fps = utils.get_video(video_path)

        first_frame = utils.get_one_frame(video, frame_num=0, current_frame=0)

        right_part_color, right_part_gray = self.board_preparator.initialize(first_frame)

        self.map_circles = utils.find_circles(right_part_gray, equalize=None, minDist=30, param1=170, param2=20, minRadius=12, maxRadius=25)
        self.map_circles = self.label_circles(self.map_circles, right_part_color)
        self.islands = self.detect_islands(right_part_color)

        current_frame_num = 0
        candidates = None
        self.left_candidates = None
        # than processing of later ones withouth unnecessary steps, just updates
        while video.isOpened():
            video.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
            ret, frame = video.read()

            if ret:
                current_frame_num += 10  # 3fps
                left_part_color, left_foreground, self.right_part_color, right_foreground = self.board_preparator.process(frame, current_frame_num)

                left_part_color = self.left_part_analyzer.process(left_part_color, left_foreground)
              
                self.right_part_color, candidates = self.update_interesting_objects(
                    right_foreground, self.right_part_color, candidates)

                #self.right_part = self.draw_circles(self.right_part_color, self.map_circles)
                self.update_view()
                #cv2.imshow("left", self.left_part_color)
                cv2.imshow("right", self.right_part_color)
                cv2.imshow("game look", np.concatenate([left_part_color, self.right_part_color], axis=1))
                #cv2.imshow("foreground", foreground)
                #cv2.imshow("stats", self.stats)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break


if __name__ == "__main__":
    board_preparator = BoardPreparator(empty_board_path="data/empty_board.jpg")
    left_part_analyzer = LeftPartAnalyzer(board_preparator)
    tracker = CycladesTracker(board_preparator, left_part_analyzer)
    tracker.run("data/cyklady_lvl1_1.mp4")
