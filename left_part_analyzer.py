import cv2
import numpy as np
import utils


class LeftPartAnalyzer():
    def __init__(self, board_preparator, debug=False):
        self.board_preparator = board_preparator
        self.gods_colors = {
            "athena": [np.array([0,0,180]), np.array([180,50,255])], 
            "ares": [np.array([0, 50, 100]), np.array([20, 255, 255])],
            "poseidon": [np.array([90, 50, 150]), np.array([100, 255, 255])], 
            "zeus": [np.array([110, 50, 150]), np.array([120, 255, 255])],
        }

        self.debug = debug
        self.objects = {}
        self.candidates = {}

        self.how_much_to_classify_as_object = 3

    def classify_pawn_by_color(self, frame):
        yellow, black, red = utils.segment_colors(frame, debug=self.debug)

        w,h, _ = frame.shape
        total = utils.calculate_total(w,h)
        if w * h < 400:
            if yellow.sum() / total > 0.3:
                return "yellow"
            elif red.sum() / total > 0.4:
                return "red"
            elif black.sum() / total > 0.4:
                return "black"
            

    def get_most_possible_god(self, cutted, total):
        ratios = []
        for name, color in self.gods_colors.items():
            mask = utils.segment_by_hsv_color(cutted, color[0], color[1])
            if self.debug:
                cv2.imshow("god mask", mask)
                cv2.waitKey(0)
            ratios.append(mask.sum() / total)

        max_god = np.argmax(ratios)
        if ratios[max_god] > 0.1:
            return list(self.gods_colors.keys())[max_god]

    def classify_gods(self, frame):
        w,h, _ = frame.shape
        if w*h > 5000:
            total = utils.calculate_total(w,h)
            god = self.get_most_possible_god(frame, total)
            return god

    def classify_cards(self, frame):
        w,h, _ = frame.shape
        if w*h > 1500 and w*h < 3000:
            return "card"

    def check_decision(self, decision, box, frame):
        if decision:
            self.board_preparator.zero_mask(box)
            self.objects[decision] = box
    
    def classify_objects(self, box, frame):
        cutted = utils.cut_obj(frame,box)

        if self.debug:
            cv2.imshow("to be detected", cutted)
            cv2.waitKey(0)

        self.check_decision(self.classify_pawn_by_color(cutted),box,frame)
        self.check_decision(self.classify_gods(cutted),box,frame)
        self.check_decision(self.classify_cards(cutted),box,frame)

    def process(self, color, foreground):
        color, self.candidates, correct_boxes = utils.update_interesting_objects(
            foreground, color, self.candidates, self.how_much_to_classify_as_object)

        for candidate_box in correct_boxes:
            self.classify_objects(candidate_box, color)

        if self.debug:
            cv2.imshow("left_color", color)
            cv2.imshow("left_fg", foreground)
            cv2.imshow("mask", self.board_preparator.mask*255)

        return self.update_view(color)

    def update_view(self, left_part_color):
        for object in self.objects.keys():
            x,y,w,h = self.objects[object]

            cv2.rectangle(left_part_color, (x, y),
                              (x + w, y + h), (0,255,0), 2)

            cv2.putText(left_part_color, object, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        return left_part_color

                