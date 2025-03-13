import cv2


class Streamer():

    def __init__(self, video_path):
        self.video_path = video_path
        self.frames = []


    def extract_frames(self, interval=1):
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("Error: Cannot open video.")
            return []
        
        # get frames rate per seconds
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for sec in range(0, int(total_frames / fps), interval):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()

            if not ret:
                print(f"Error: Failed to capture frame at second {sec}")
                break

            self.frames.append(frame)

        cap.release()
