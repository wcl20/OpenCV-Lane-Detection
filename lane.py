import cv2
import numpy as np

def get_edge(image):
    # Gray Scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny Edge Detection
    return cv2.Canny(blur, 50, 150)

def get_region(image, vertices):
    # Create mask
    mask = np.zeros_like(image)
    # Fill region white
    cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), 255)
    # Return masked image
    return cv2.bitwise_and(image, mask)

def get_lines(image):
    # Hough Transform
    lines = cv2.HoughLinesP(image, 2, np.pi / 180, 15, np.array([]), minLineLength=20, maxLineGap=2)
    # return lines
    # Calculate average line for left and right lane
    parameters = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # Compute slope and intersection
            parameters.append(np.polyfit((x1, x2), (y1, y2), 1))
    parameters = np.array(parameters)
    # Right lane has positive slope
    right = np.average(parameters[parameters[:, 0] > 0], axis=0)
    # Left lane has negative slope
    left = np.average(parameters[parameters[:, 0] < 0], axis=0)
    # Create line from line parameters
    return np.array([create_line(image, left), create_line(image, right)])

def create_line(image, parameters):
    height, width = image.shape
    m, c = parameters
    # Define y coordinates
    y1, y2 = height, int(height * 0.7)
    # Compute x coordinates
    x1, x2 = int((y1 - c) / m), int((y2 - c) / m)
    return np.array([x1, y1, x2, y2])

def draw_lines(image, lines):
    # Create an image with lines
    mask = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(mask, (x1, y1), (x2, y2), (255, 0, 0), 5)
    # Combine original image with line image
    return cv2.addWeighted(image, 0.5, mask, 1, 1)
