import cv2
import numpy as np
import utils

class RightPartAnalyzer():
    def __init__(self,debug=False):
        self.debug = debug

        self.objects = {}
        self.candidates = {}

        self.moved = False
        self.placed = False
        self.waiting_moved = None
        self.waiting_placed = None 

        self.how_much_to_classify_as_object = 3

        self.cities = {'yellow': 0, 'red': 0, 'black': 0}
        self.repeat_city_assignment = []


    def analyze_map(self, right_part_color, right_part_gray):
        self.map_circles = utils.find_circles(right_part_gray, equalize=None, minDist=30, param1=170, param2=20, minRadius=12, maxRadius=25)
        self.map_circles = self.label_circles(self.map_circles, right_part_color)
        self.islands = self.detect_islands(right_part_color)

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

    def object_color(self, yellow, black, red, blue=None, violet=None, gray=None, orange=None):   
        if blue is not None and blue.sum() > yellow.sum() and blue.sum() > black.sum() and blue.sum() > red.sum():
            if blue.sum() > violet.sum() and blue.sum() > gray.sum() and blue.sum() > orange.sum():
                return "blue"
        if gray is not None and gray.sum() > yellow.sum() and gray.sum() > black.sum() and gray.sum() > red.sum():
            if gray.sum() > blue.sum() and gray.sum() > violet.sum() and gray.sum() > orange.sum() and gray.sum() > gray.size * 0.9:
                return "gray"
        if orange is not None and orange.sum() > yellow.sum() and orange.sum() > black.sum() and orange.sum() > red.sum():
            if orange.sum() > blue.sum() and orange.sum() > violet.sum() and orange.sum() > gray.sum() and orange.sum() > orange.size * 0.9:
                return "orange"
        if violet is not None and violet.sum() > black.sum():
            return "violet"
        elif red.sum() > red.size * 0.8:
            return "red"
        elif yellow.sum() > yellow.size * 0.8 and yellow.sum() > black.sum():
            return "yellow"
        elif black.sum() > black.size * 0.8:
            return "black"
        else:
            return "unknown"

    def is_moved(self, name, frame):
        color = name.split(" ")[0]
        if name not in self.objects.keys(): 
            return -1

        for i, obj_box in enumerate(self.objects[name]):
            cutted= utils.cut_obj(frame, obj_box)
            if self.debug:
                cv2.imshow("moved?", cutted)
            yellow, black, red = utils.segment_colors(cutted, debug=self.debug)
            new_color = self.object_color(yellow, black, red)
            if new_color != color:
                print(color, new_color)
                return i

        return -1

    def process(self, color, foreground, draw_circles = False):
        color, self.candidates, correct_boxes = utils.update_interesting_objects(
            foreground, color, self.candidates, self.how_much_to_classify_as_object)

        for candidate_box in correct_boxes:
            self.classify_objects(candidate_box, color)

        if self.debug:
            cv2.imshow("right_color", color)
            cv2.imshow("right_fg", foreground)

        if draw_circles:
            color= self.draw_circles(color, self.map_circles)

        return self.update_view(color)
   
    def city_island_mapping(self, x, y, w, h, frame):
        current_island = self.detect_current_island(x, y, w, h, frame)
        if current_island is not None:
            xr, yr, wr, hr = cv2.boundingRect(current_island[0])
            if len(self.islands[(xr, yr, wr, hr)]) > 1:
                owner = self.islands[(xr, yr, wr, hr)][1]
                self.cities[owner] += 1
            else:
                self.repeat_city_assignment.append([xr, yr, wr, hr, frame])
    
    def island_object_mapping(self, x, y, w, h, frame, color):
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
    
    def classify_objects(self, box, frame):
        x, y, w, h = box
        cutted = frame[y:y+h, x:x+w]

        if self.debug:
            cv2.imshow("is_object?", cutted)

        yellow, black, red = utils.segment_colors(cutted, debug=self.debug)
        blue, violet, gray, orange = utils.segment_colors_cities(cutted)

        color = self.object_color(
            yellow, black, red, blue, violet, gray, orange)
        obj_type = self.object_type(x, y, w, h, frame)
        if color in ['blue', 'violet', 'gray', 'orange']:
            if w*h<350 or  obj_type != 'warrior':
                return
            obj_type = 'city'
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
            self.island_object_mapping(x, y, w, h, frame, color)

        if obj_type != 'city' and self.is_moved(name, frame) != -1:
            self.objects[name].pop(self.is_moved(name,frame))
            self.moved = True

        if not name in self.objects.keys():
            self.objects[name] = [[x, y, w, h]]
            self.placed = True
        else:
            self.objects[name].append([x, y, w, h])
            self.placed = True

        if obj_type == "city":
            self.city_island_mapping(x, y, w, h, frame)
        for i in range(len(self.repeat_city_assignment)):
            params = self.repeat_city_assignment[i]
            self.city_island_mapping(params[0], params[1], params[2], params[3], params[4])
            self.repeat_city_assignment.pop(i)

    def update_view(self, frame):
        h = 0
        right_stats = np.zeros(frame.shape, dtype=np.uint8)
        right_stats.fill(255)
        for key, l in self.objects.items():
            cv2.putText(right_stats, key + ": " + str(len(l)), (20,
                        20 + 20 * h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            h += 1

        taken_islands = sum(
            [1 for island, stats in self.islands.items() if len(stats) > 1])
        color_islands = {'red': 0, 'black': 0, 'yellow': 0}
        for island in self.islands.values():
            if len(island) > 1:
                if island[1] in ['blue', 'violet', 'gray', 'orange']:
                    continue
                color_islands[island[1]] += 1
        for color, cnt in color_islands.items():
            cv2.putText(right_stats, color + " islands: " + str(cnt), (20,
                                                                      20 + 20 * h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            h += 1
        cv2.putText(right_stats, "taken islands: " + str(taken_islands), (20,
                                                                         20 + 20 * h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        h += 1
        if self.moved and self.waiting_moved is None:
            self.waiting_moved = 9
        elif self.moved:
            self.waiting_moved -= 1
            if self.waiting_moved == 0:
                self.moved = False
                self.waiting_moved = None
        cv2.putText(right_stats, "moved_counter: " + str(self.moved), (20,
                                                                   20 + 20 * h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        h += 1

        for owner, cnt in self.cities.items():
            cv2.putText(right_stats, owner + " cities: " + str(cnt), (20,
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
            cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
            cv2.putText(frame, "island", (xr, yr),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, utils.get_font_color(color), 2)

        for object_ in self.objects.keys():
            for x, y, w, h in self.objects[object_]:
                color, obj_type = object_.split(" ")
                cv2.rectangle(frame, (x, y),
                              (x + w, y + h), utils.get_font_color(color), 2)
                cv2.putText(frame, obj_type, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, utils.get_font_color(color), 2)

        return frame, right_stats
