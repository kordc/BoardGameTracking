import cv2
import numpy as np
import utils
from collections import defaultdict


class LeftPartAnalyzer():
    def __init__(self, board_preparator, debug=False):
        self.board_preparator = board_preparator
        self.gods_colors = {
            "athena": [np.array([0, 0, 180]), np.array([180, 50, 255])],
            "ares": [np.array([0, 50, 100]), np.array([20, 255, 255])],
            "poseidon": [np.array([90, 50, 150]), np.array([100, 255, 255])],
            "zeus": [np.array([110, 50, 150]), np.array([120, 255, 255])],
        }

        self.debug = debug
        self.single_objects = {}
        self.multi_objects = defaultdict(lambda: [])
        self.candidates = {}
        self.how_much_to_classify_as_object = 3

    def classify_pawn_by_color(self, frame, box):
        # Get the most possible pawn by color
        yellow, black, red = utils.segment_colors(frame, debug=self.debug)

        w, h, _ = frame.shape

        total = utils.calculate_total(w, h)
        _, y, _, _ = box

        # Pawn cannot be at the bottom of the board
        # if y > 600:
        #     return

        if w * h < 400:
            if yellow.sum() / total > 0.4:
                return "yellow"
            elif red.sum() / total > 0.4:
                return "red"
            elif black.sum() / total > 0.4:
                return "black"

    def get_most_possible_god(self, cutted, total):
        # Get the most possible god by color
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
        # Classify gods
        h, w, _ = frame.shape
        if w*h > 5000 and w/h > 1.5:
            total = utils.calculate_total(w, h)
            god = self.get_most_possible_god(frame, total)
            return god

    def classify_cards(self, frame):
        # Classify cards
        if len(self.multi_objects["card"]) >= 3:
            return

        h, w, _ = frame.shape
        if w*h > 500 and w*h < 3000 and w/h > 0.8 and w/h < 1.2:
            return "card"

    def check_decision(self, decision, box, frame):
        # Check if decision is valid and update objects
        if decision:
            self.board_preparator.zero_mask(box)
            if decision == "card":
                self.multi_objects[decision].append(box)
            else:
                self.single_objects[decision] = [box]

    def classify_objects(self, box, frame):
        # Classify objects by color and type
        cutted = utils.cut_obj(frame, box)

        if self.debug:
            cv2.imshow("to be detected", cutted)
            cv2.waitKey(0)

        self.check_decision(
            self.classify_pawn_by_color(cutted, box), box, frame)
        self.check_decision(self.classify_gods(cutted), box, frame)
        self.check_decision(self.classify_cards(cutted), box, frame)

    def process(self, color, foreground):
        # This is the main function that is called for each frame
        color, self.candidates, correct_boxes = utils.update_interesting_objects(
            foreground, color, self.candidates, self.how_much_to_classify_as_object, debug_contours=self.debug)

        for candidate_box in correct_boxes:
            self.classify_objects(candidate_box, color)

        if self.debug:
            cv2.imshow("left_color", color)
            cv2.imshow("left_fg", foreground)
            cv2.imshow("mask", self.board_preparator.mask*255)

        return self.update_view(color)

    def update_view(self, left_part_color):
        # Update labeling, texts and bounding boxes
        stats = np.full((200, 500, 3), 255, dtype=np.uint8)

        for i, god in enumerate(self.gods_colors.keys()):
            if god in self.single_objects:
                for pawn in ["yellow", "red", "black"]:
                    if pawn in self.single_objects:
                        if np.abs(self.single_objects[god][0][1] - self.single_objects[pawn][0][1]) < 20:
                            cv2.putText(stats, f"{pawn} player has {god}", (50, 30*i),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        cv2.putText(stats, f"number of cards: {len(self.multi_objects['card'])}", (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        for objects in [self.single_objects, self.multi_objects]:
            for object_type in objects.keys():
                for object in objects[object_type]:
                    x, y, w, h = object

                    cv2.rectangle(left_part_color, (x, y),
                                  (x + w, y + h), (0, 255, 0), 1)

                    cv2.putText(left_part_color, object_type, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("gods stats", stats)
        cv2.moveWindow("gods stats", 0, 0)

        return left_part_color
