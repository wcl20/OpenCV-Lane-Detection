import cv2
import numpy as np
from lane import get_edge, get_region, get_lines, draw_lines

def main():
    # Read Image
    cap = cv2.VideoCapture("video/video.mp4")
    while (cap.isOpened()):
        # Capture frame
        ret, frame = cap.read()
        if ret:
            # Canny Edge Detection
            canny = get_edge(frame)
            # Region of interest
            height, width, _ = frame.shape
            vertices = [ (0, height), (width, height), (width // 2, height * 0.6) ]
            region = get_region(canny, vertices)
            # # Get Lane Lines
            lines = get_lines(region)
            # # Draw lines on frame
            output = draw_lines(frame, lines)
            # Display frame
            cv2.imshow("Lane Detection", output)
            # Press 'q' to quit video capture
            if cv2.waitKey(1) & 0xFF == ord('q'):
        	       break
        else:
            break
    # Release video capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
