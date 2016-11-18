import cv2
import dlib
import numpy as np

import sys
PREDICTOR_PATH = ".\shape_predictor_68_face_landmarks.dat"
IMAGE_PATH = "base.jpg"
POINTS_PATH = "base.txt"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

    
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True
    
    
def getLandmarks(img):

    rects = detector(img, 1)
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    landmarkMatrix = np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])
    
    # Convert matrix to array
    points = np.squeeze(np.asarray(landmarkMatrix))
    
    # # Add boundary points for delaunay triangulation
    # size = img.shape
    # w = size[1]
    # h = size[0]
    
    # boundaryPts = np.array([ [0,0], [w/2,0], [w-1,0], [w-1,h/2], [ w-1, h-1 ], [ w/2, h-1 ], [0, h-1], [0,h/2] ])
    # points = np.concatenate((points, boundaryPts), axis=0)

    return points

def savePoints(points):
    f = open(POINTS_PATH, "w")
    for p in points:
        f.write("%s %s\n" % (p[0], p[1]))


if __name__ == '__main__':
    # Read Images
    img = cv2.imread(IMAGE_PATH)
    points = getLandmarks(img)
    savePoints(points)
    