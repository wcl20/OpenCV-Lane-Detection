import cv2
import numpy as np
from lane import get_edge, get_region, get_lines, draw_lines

def main():
    # Read Image
    image = cv2.imread("image/image.jpg")
    # Canny Edge Detection
    canny = get_edge(image)
    # Region of interest
    height, width, _ = image.shape
    vertices = [ (0, height), (width, height), (width // 2, height * 0.6) ]
    region = get_region(canny, vertices)
    # Get Lane Lines
    lines = get_lines(region)
    # Draw lines on image
    output = draw_lines(image, lines)
    # Display Image
    cv2.imshow("Lane Detection", output)
    cv2.waitKey(5000)

if __name__ == '__main__':
    main()
