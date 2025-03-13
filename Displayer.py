import cv2
import numpy as np
import time

class Displayer():
    def __init__(self,blur_ksize=(5, 5)):
        self.blur_ksize = blur_ksize
        pass

    def __blur_contours(self, frame, contours):
      
        blurred_frame = frame.copy()
        
        # Create an empty mask to hold the contours
        mask = np.zeros_like(frame)
         
        # Fill the mask with white color inside the contour regions
        for c in contours:
            # cv2.rectangle(mask, (c.x, c.y), (c.x + c.w, c.y + c.h), (0, 255, 0), thickness=cv2.FILLED)
            cv2.drawContours(mask, [c.to_contour()], -1, (255, 255, 255), thickness=cv2.FILLED)
        
        # Blur the entire frame first
        blurred = cv2.GaussianBlur(mask, self.blur_ksize, 0)
        
        # Mask the blurred image with the original contours
        blurred_frame = cv2.bitwise_and(blurred, mask)  # Apply blur to the contour areas only
        
        # Mask the original image with the inverse of the contours (keep other areas sharp)
        non_blurred_frame = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
        
        # Combine both parts: the blurred contours and the non-blurred background
        final_frame = cv2.add(non_blurred_frame, blurred_frame)
        
        return final_frame

    def blurCountrousFramesPlusAddTime(self, metaFrameList):
        for i, cf in enumerate(metaFrameList):
            newFrame = self.__blur_contours(cf.frame, cf.contourObjectList) 
            
            # Adding timestamp to the frame
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')  # Current system time
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(newFrame, timestamp, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow(f'Motion Frame with Bounding Boxes {i+1}', newFrame)
            cv2.waitKey(0)
        cv2.destroyAllWindows()    


    def draw_bounding_boxes(self,metaFrameList):
        for i, cf in enumerate(metaFrameList):
            contours = cf.contourObjectList
            frame = cf.frame
            for c in contours:
                # ציור מלבן על האזור שבו התגלתה תנועה
                cv2.rectangle(frame, (c.x, c.y), (c.x + c.w, c.y + c.h), (0, 255, 0), 2)
                
            cv2.imshow(f'Motion Frame with Bounding Boxes {i+1}', frame)
            cv2.waitKey(0)
        cv2.destroyAllWindows()    

   
            

