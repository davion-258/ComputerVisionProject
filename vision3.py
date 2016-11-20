# Project: Generate baby face image from his father and mother images
# Developed by
#       1412334: Le Hoang Nam
#       1412669: Ngo Huynh Ngoc Khanh
# Reference :
# 

import os
import cv2
import numpy as np
import dlib
import sys

PREDICTOR_PATH = ".\shape_predictor_68_face_landmarks.dat"
IMAGE1_PATH = "black1.png"
IMAGE2_PATH = "test1.jpg"
IMAGE_BASE_PATH = "base.jpg"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Return a numpy array of landmark points
def determineLandmarkPoints(img) :
    
    rects = detector(img, 1)
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    landmarkMatrix = np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])
    
    # Convert matrix to array
    points = np.squeeze(np.asarray(landmarkMatrix))

    return points


# def averagePoints(points1, points2, pointsBase) :
    # for i in xrange(0, len(points1)):
        # x = (1 - beta) * ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0] + beta * pointsBase
        # y = (1 - beta) * ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1] + beta * pointsBase
        # points.append((int(x), int(y)))
        
def averagePoints(points1, points2, alpha) :

    points = []
    for i in xrange(0, len(points1)):
        x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
        y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
        points.append((int(x), int(y)))
        
    return points
    
    
# Check if a point is inside a rectangle
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

    
def calculateTriangles(size, points) :
    rect = (0, 0, size[1], size[0])

    # Create subdiv
    subdiv = cv2.Subdiv2D(rect);
    
    # Insert points into subdiv
    for p in points:
        subdiv.insert(p) 
    
    triangleList = subdiv.getTriangleList();
    
    delaunayTri = []
    
    pt = []    
    
    count= 0    
    
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            count = count + 1 
            ind = []
            for j in xrange(0, 3):
                for k in xrange(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)                            
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        
        pt = []        
            
    
    return delaunayTri
        
def toListTuples(array) :
    points = []
    for line in array :
        points.append((int(line[0]), int(line[1])))
        
    return points
    
    
if __name__ == '__main__' :
    
    # Read source images
    img1 = cv2.imread(IMAGE1_PATH)
    # img2 = cv2.imread(IMAGE2_PATH)
    # imgBase = cv2.imread(IMAGE_BASE_PATH)
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    copy = img1.copy()
    
    # Determine Facial Landmark Points
    points1 = determineLandmarkPoints(img1)
    # points2 = determineLandmarkPoints(img2)
    # points = determineLandmarkPoints(imgBase)
    
    points = toListTuples(points1)
    
    hullIndex = cv2.convexHull(np.array(points), returnPoints = False)
    
    hull = []
    
    for i in xrange(0, len(hullIndex)):
        hull.append(points[hullIndex[i]])

    # Calculate Rectangle
    x, y, w, h = cv2.boundingRect(np.float32([hull]))
    
    print img1[x + w/2][y + h/2]
    
    # cv2.rectangle(copy, r, pt2, color[, thickness[, lineType[, shift]]])
    
    # print points1
    # Ratio between mom and dad
    # alpha = 0.5
    
    # Ratio between morph image and example baby face
    # beta = 0.5
    
    # Compute weighted average point coordinates
    # points = averagePoints(points1, points2, alpha)
    
    
    # Calculate Delaunay triangles for output image
    # triangles = calculateTriangles(img1.shape, points1)
    
    # print triangles
    # # Allocate space for final output
    # imgMorph = np.zeros(img1.shape, dtype = img1.dtype)
    
    # # Morph in each triangles
    # for indice in triangles :
        # x,y,z = indice

        # # Rounding
        # x = int(x)
        # y = int(y)
        # z = int(z)
        
        # # Get Triangle Coordinate
        # triangle1 = [points1[x], points1[y], points1[z]]
        # triangle2 = [points2[x], points2[y], points2[z]]
        # triangle = [ points[x], points[y], points[z] ]

        # # Morph one triangle at a time.
        # morphTriangle(img1, img2, imgBase, triangle1, triangle2, triangle, alpha)
        
        
    # # Display result
    # cv2.imshow("Image 1", copy)
    # cv2.imshow("Image 2", img2)
    # cv2.imshow("Result", result)
    cv2.waitKey(0)