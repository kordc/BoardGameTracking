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
     
def centres_within(boxes1, boxes2, sigma = 10):
    #This calculates if centres of boxes are withing some threshold
    xc1, yc1 = calculate_means(boxes1)
    xc2, yc2 = calculate_means(boxes2)

    x_diff = np.abs(xc1 - np.transpose(xc2))
    y_diff = np.abs(yc1 - np.transpose(yc2))

    total_diff = x_diff + y_diff
    
    similar = np.argwhere(total_diff < sigma)
    return similar
    