from Detector import Detector
from Streamer import Streamer
from Displayer import Displayer


video_path = r'C:\Users\reutb\Downloads\hand.mp4'
streamer = Streamer(video_path)
streamer.extract_frames()
 
motion_detector = Detector(streamer.frames)
motion_detector.detect_motion()

print(f"Detected motion in {len(motion_detector.motion_frames)} frames.")

displayer = Displayer()
 
displayer.draw_bounding_boxes(motion_detector.motionFrameWithMetaList)

displayer.blurCountrousFramesPlusAddTime(motion_detector.motionFrameWithMetaList)

 
