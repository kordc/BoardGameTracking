import cv2
import numpy as np
import utils
import argparse

from board_preparator import BoardPreparator
from left_part_analyzer import LeftPartAnalyzer
from right_part_analyzer import RightPartAnalyzer
class CycladesTracker:
    def __init__(self, board_preparator: BoardPreparator, left_part_analyzer: LeftPartAnalyzer, right_part_analyzer: RightPartAnalyzer):
        self.board_preparator = board_preparator
        self.left_part_analyzer = left_part_analyzer
        self.right_part_analyzer = right_part_analyzer

        
        self.create_and_move("stats",0,300)
        # self.create_and_move("foreground",500,500)
        self.create_and_move("game look", 500,0)
    def create_and_move(self,name, x,y):
        cv2.namedWindow(name)
        cv2.moveWindow(name, x,y)

    def run(self, video_path, starting_frame):
        # At first processing of the first frame
        video, width, height, fps = utils.get_video(video_path)
        first_frame = utils.get_one_frame(video, frame_num=starting_frame, current_frame=starting_frame)
        
        right_part_color, right_part_gray = self.board_preparator.initialize(first_frame)
        self.right_part_analyzer.analyze_map(right_part_color, right_part_gray)

        resized = cv2.resize(first_frame, None, fx=0.4, fy=0.4)
        output = cv2.VideoWriter('results/output_cyklady_lvl3_1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                 resized.shape[:2][::-1])
        current_frame_num = starting_frame
        while video.isOpened():
            video.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
            ret, frame = video.read()
            if ret:
                current_frame_num += 10  # 3fps
                left_part_color, left_foreground, right_part_color, right_foreground = self.board_preparator.process(frame, current_frame_num)

                left_part_color = self.left_part_analyzer.process(left_part_color, left_foreground)
                right_part_color, right_stats = self.right_part_analyzer.process(right_part_color, right_foreground)


                # cv2.imshow("foreground", np.concatenate([left_foreground, right_foreground], axis=1))
                game_look = np.concatenate(
                    [left_part_color, right_part_color], axis=1)
                cv2.imshow("game look", game_look)
                cv2.imshow("stats", right_stats)
                for _ in range(10):
                    output.write(game_look)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        output.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                            prog = 'CycladeTracker',
                            description = 'Program that tracks cyclade board game',
                            epilog = 'have fun!')

    parser.add_argument('-f','--filename', default="data/cyklady_lvl3_1.mp4", help="path to video")
    parser.add_argument('-e', "--empty_board_image", default="data/empty_board.jpg", help="path to image of an empty board")
    parser.add_argument('-db','--debug_board', default=False, help = "Set to True if you want to see debug images of the board")
    parser.add_argument('-dl','--debug_left', default=False, help = "Set to True if you want to see debug images of the left side ")
    parser.add_argument('-dr', '--debug_right', default=False, help = "Set to True if you want to see debug images of the right side")
    parser.add_argument('-sf', '--starting_frame', default=0, help = "Set starting frame")
    
    args = parser.parse_args()

    board_preparator = BoardPreparator(empty_board_path=args.empty_board_image, debug=args.debug_board)
    left_part_analyzer = LeftPartAnalyzer(board_preparator,debug=args.debug_left)
    right_part_analyzer = RightPartAnalyzer(debug = args.debug_right)

    tracker = CycladesTracker(board_preparator, left_part_analyzer, right_part_analyzer)
    tracker.run(args.filename, int(args.starting_frame))
