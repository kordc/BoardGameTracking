import cv2
import numpy as np
import utils

from board_preparator import BoardPreparator
from left_part_analyzer import LeftPartAnalyzer
from right_part_analyzer import RightPartAnalyzer
class CycladesTracker:
    def __init__(self, board_preparator: BoardPreparator, left_part_analyzer: LeftPartAnalyzer, right_part_analyzer: RightPartAnalyzer):
        self.board_preparator = board_preparator
        self.left_part_analyzer = left_part_analyzer
        self.right_part_analyzer = right_part_analyzer

    def run(self, video_path):
        # At first processing of the first frame
        video, width, height, fps = utils.get_video(video_path)

        first_frame = utils.get_one_frame(video, frame_num=0, current_frame=0)

        right_part_color, right_part_gray = self.board_preparator.initialize(first_frame)
        self.right_part_analyzer.analyze_map(right_part_color, right_part_gray)

        current_frame_num = 0
        while video.isOpened():
            video.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
            ret, frame = video.read()

            if ret:
                current_frame_num += 10  # 3fps
                left_part_color, left_foreground, right_part_color, right_foreground = self.board_preparator.process(frame, current_frame_num)

                left_part_color = self.left_part_analyzer.process(left_part_color, left_foreground)
                right_part_color, right_stats = self.right_part_analyzer.process(right_part_color, right_foreground)

                #cv2.imshow("left", self.left_part_color)
                #cv2.imshow("right", self.right_part_color)

                cv2.imshow("game look", np.concatenate([left_part_color, right_part_color], axis=1))
                #cv2.imshow("foreground", foreground)
                cv2.imshow("stats", right_stats)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break


if __name__ == "__main__":
    board_preparator = BoardPreparator(empty_board_path="data/empty_board.jpg")
    left_part_analyzer = LeftPartAnalyzer(board_preparator)
    tracker = CycladesTracker(board_preparator, left_part_analyzer)
    tracker.run("data/cyklady_lvl1_1.mp4")
