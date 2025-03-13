import cv2
import imutils
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ContourObject():
    x: int
    y: int
    w: int
    h: int

    def to_contour(self):
        """ Convert a ContourObject to an OpenCV contour (list of points). """
        # Generate points for the bounding box of the contour
        contour = np.array([
            [self.x, self.y],  # Top-left corner
            [self.x + self.w, self.y],  # Top-right corner
            [self.x + self.w, self.y + self.h],  # Bottom-right corner
            [self.x, self.y + self.h]  # Bottom-left corner
        ], dtype=np.int32)
        
        # Close the contour (the last point is the same as the first)
        return contour.reshape((-1, 1, 2))
    
@dataclass(frozen=True)
class FrameMeta():
    frame: any
    contourObjectList : List[ContourObject]
 
  
class Detector():
    def __init__(self, frames, min_area=3000):
        self.frames = frames
        self.motion_frames = []  
        self.thresholded_frames = []  
        self.min_area = min_area  
        self.motionFrameWithMetaList : List[FrameMeta] = []

    def detect_motion(self):
        if len(self.frames) < 2:
            print("Not enough frames for motion detection.")
            return
        
        firstFrame = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2GRAY)
        
        for i in range(1, len(self.frames)):
            frame = self.frames[i]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # check diff between frames
            frameDelta = cv2.absdiff(firstFrame, gray)
            
            # 85 is less sensetive
            thresh = cv2.threshold(frameDelta, 85, 255, cv2.THRESH_BINARY)[1]
            
            # find the changes in pic, no need more than one
            thresh = cv2.dilate(thresh, None, iterations=1)
            
            # find countours
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            contour_objects  = []
            for c in cnts:
                if cv2.contourArea(c) < self.min_area:
                    continue

                #motion detected
                (x, y, w, h) = cv2.boundingRect(c)
                contour_objects.append(ContourObject(x, y, w, h))
        
            # kepp motion frame
            if contour_objects:
                self.motionFrameWithMetaList.append(FrameMeta(frame,contour_objects))
                self.motion_frames.append(frame)
                self.thresholded_frames.append(thresh)
                
