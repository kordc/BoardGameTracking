import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt

def find_circles(frame, equalize="local", blur=False, **kwargs):
    cframe = frame.copy()
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur:
        frame = cv2.GaussianBlur(frame, (3,3), 0)
    if equalize == "global":
        frame = cv2.equalizeHist(frame)
    elif equalize == "local":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        frame = clahe.apply(frame)

    circles = cv2.HoughCircles(
        frame, cv2.HOUGH_GRADIENT, 1, **kwargs)
    circles = np.uint16(np.around(circles))
    return circles

def plot_circles(frame, circles):
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    return frame


def get_video(path):
    video = cv2.VideoCapture(path)
    if video.isOpened():
        print('Video loaded')
    video_width = int(video.get(3))
    video_height = int(video.get(4))

    print(video_height, video_width)

    video_fps = video.get(cv2.CAP_PROP_FPS)
    print(video_fps)

    return video, video_width, video_height, video_fps


def get_one_frame(video, frame_num, current_frame):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
    res, frame = video.read()
    video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    return frame

def detect_edges(frame, equalize=True, blur=False, low_threshold=70, high_threshold=190):
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur:
        frame = cv2.GaussianBlur(frame, (3,3), 0)
        frame = cv2.GaussianBlur(frame, (3,3), 0)
    if equalize:
        frame = cv2.equalizeHist(frame)
    edges = cv2.Canny(frame, low_threshold, high_threshold, apertureSize=3)
    return edges

def find_circles(frame, equalize="local", blur=False, **kwargs):
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur:
        frame = cv2.GaussianBlur(frame, (3,3), 0)
    if equalize == "global":
        frame = cv2.equalizeHist(frame)
    elif equalize == "local":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        frame = clahe.apply(frame)

    circles = cv2.HoughCircles(
        frame, cv2.HOUGH_GRADIENT, 1, **kwargs)
    circles = np.uint16(np.around(circles))

    return circles



def plot_colors(hsv1, hsv2):
    square1 = np.full((10, 10, 3), hsv1, dtype=np.uint8) / 255.0
    square2 = np.full((10, 10, 3), hsv2, dtype=np.uint8) / 255.0
    plt.imshow(hsv_to_rgb(np.concatenate([square1, square2],axis=0)))

def segment_by_hsv_color(frame, lower, upper, plot_colors=False):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if plot_colors:
        plot_colors(lower, upper)
    
    mask = cv2.inRange(hsv, lower, upper)

    return mask

def calculate_means(boxes):
    #calculates center of the bounding boxes
    return [boxes[:, 0] + boxes[:,2]//2], [boxes[:, 1] + boxes[:,3]//2]

def calculate_area(boxes):
    return [boxes[:,2] * boxes[:,3]]

def centres_within(boxes1, boxes2, sigma_center = 10, sigma_area = 100):
    #This calculates if centres of boxes are withing some threshold
    xc1, yc1 = calculate_means(boxes1)
    xc2, yc2 = calculate_means(boxes2)

    x_diff = np.abs(xc1 - np.transpose(xc2))
    y_diff = np.abs(yc1 - np.transpose(yc2))
    
    center_diff = x_diff + y_diff

    #area_diff = np.abs(calculate_area(boxes1) - np.transpose(calculate_area(boxes2))) #! not checked
    
    similar = np.argwhere((center_diff < sigma_center)) #& (area_diff < sigma_area))
    return similar

def get_boxes(boxes):
    x11, y11, w, h = np.split(boxes, 4, axis=1)
    return x11, y11, x11+w , y11+h

def iou(boxes2,boxes1):
    #!One way to see if boxes are the same during iterations.
    #But actually this may be very sensitive to some big objects showing up
    
    x11, y11, x12, y12 = get_boxes(boxes1)
    x21, y21, x22, y22 = get_boxes(boxes2) # from num_of_boxes by 4 matrix we go to 4 vectors.
    xA = np.maximum(x11, np.transpose(x21)) #transpose is done to make it broadcastable and has this maximum over all boxes!
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return np.argwhere(iou > 0.5)
    
def calculate_total(w, h):
        return w * h * 255

def segment_colors(frame, debug=False):
    yellow = segment_by_hsv_color(frame, np.array([20, 100, 100]), np.array([30, 255, 255]))
    black = segment_by_hsv_color(frame, np.array([0, 0, 0]), np.array([180, 255, 30]))
    red = segment_by_hsv_color(frame, np.array([0, 100, 100]), np.array([10, 255, 255]))

    if debug:
        cv2.imshow("segmented_colors", np.concatenate([yellow,black,red], axis=1))
        cv2.waitKey(0)
    return yellow, black, red

def cut_obj(frame, box):
    x, y, w, h = box
    return frame[y:y+h, x:x+w]

def get_land_mask(frame, label_circles=False):
    if label_circles:
        return segment_by_hsv_color(frame, np.array([10, 50, 50]), np.array([49, 255, 255]))
    else:
        return segment_by_hsv_color(frame, np.array([50, 50, 50]), np.array([70, 255, 255]))

def get_sea_mask(frame, label_circles=False):
    if label_circles:
        return segment_by_hsv_color(frame, np.array([50, 50, 50]), np.array([110, 255, 255]))
    else:
        return segment_by_hsv_color(frame, np.array([100, 100, 100]), np.array([140, 255, 255]))

def update_interesting_objects(foreground, frame, candidates, being_seen_limit, left=False, debug_contours = False):
        # bases on foreground it updates interesting objects
        cnts, hier = cv2.findContours(
            foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes, correct_boxes = [], [] # boxes found and boxes containing actual object
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)

            #For debugging purposes
            # if debug_contours:
            #     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)

            #neglect very small boxes and also very big
            if w*h > 80 and w*h < 10000: 
                boxes.append([x,y,w,h]) # Getting coordinates of every new found box
                
               
       
        #there is no valid box to be processed further
        if len(boxes) == 0:
            return frame, candidates, correct_boxes

        if not candidates: # if this is first iteration
            candidates = {tuple(box): 1 for box in boxes}
        else:
            old = list(candidates.keys()) #previous candidates
            new = boxes # currently found boxes
            matches = centres_within(np.array(old), np.array(new)) # which old box match to which new box
            new_candidates = {tuple(box): 1 for box in boxes}
            for match in matches: # Box ith from previous iteration matched to jth from this iteration
                i,j = match
                #If there is a match we increase counter of old candidate by 1
                new_candidates[tuple(new[i])] = candidates[old[j]] + 1
                #! This equality here may be problematic as sometimes more than one box can be matched!
                # If box was seen for few times we check if it contains an object
                if new_candidates[tuple(new[i])] == being_seen_limit:
                    if debug_contours:
                        cv2.rectangle(frame, (x, y), (x+w, y+h),(255, 0, 0), 2)

                    correct_boxes.append(new[i])
                     
            candidates = new_candidates
        return frame, candidates, correct_boxes

def get_font_color(color):
    if color == "yellow":
        font_color = (0, 255, 255)
    elif color == "black":
        font_color = (0, 0, 0)
    elif color == "red":
        font_color = (0, 0, 255)
    else:
        font_color = (255, 255, 255)
    return font_color