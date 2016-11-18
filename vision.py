# Project: Generate baby face image from his father and mother images
# Developed by
#       1412334: Le Hoang Nam
#       1412669: Ngo Huynh Ngoc Khanh
# Reference : http://www.learnopencv.com/

import os
import cv2
import numpy as np
import dlib
import sys

PREDICTOR_PATH = ".\shape_predictor_68_face_landmarks.dat"
IMAGE1_PATH = "test1.jpg"
IMAGE2_PATH = "test2.jpg"
IMAGE_BASE_PATH = "base.jpg"

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


def toListTuples(array) :
    points = []
    for line in array :
        points.append((int(line[0]), int(line[1])))
        
    return points
    
    
def getTriangles(size, points):
    # Rectangle to be used with Subdiv2D
    rect = (0, 0, size[1], size[0])
    
    # Create an instance of Subdiv2D
    # ali: boundary to add points -> otherwise: raise exception
    subdiv = cv2.Subdiv2D(rect);
    
    # Insert landmarks points into subdiv
    for point in points:
        subdiv.insert(point)
        
    triangleList = subdiv.getTriangleList()
    
    # Convert to indices
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
    
    
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True
    
    
def drawDelaunay(triangleList, img):
    delaunay_color = (255,0,0)
    size = img.shape
    r = (0, 0, size[1], size[0])
    for t in triangleList :
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.CV_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.CV_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.CV_AA, 0)

            
# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, imgB, img, t1, t2, t, alpha, beta) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in xrange(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    imgBRect = imgB[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)
    warpImageB  = applyAffineTransform(imgBRect, tRect, tRect, size)
    

    # Alpha blend rectangular patches
    imgRect = (1 - beta) * ((1.0 - alpha) * warpImage1 + alpha * warpImage2) + beta * warpImageB

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask            

    

# def getMorphedResultImage(img1, img2, points1, points2, points, alpha):

    # imgMorph = img2
    
    # # Determine Delaunay Triangles of imgMorph
    # triangles = getTriangles(imgMorph.shape, points)
    
    
    # # Morph 2 images
    # morph(img1, img2, imgMorph, triangles, points1, points2, points, alpha)
    
    # return imgMorph
    
def readPoints(path):
    # Create an array of points.
    points = [];
    
    # Read points
    with open(path) as file :
        for line in file :
            x, y = line.split()
            points.append([int(x), int(y)])
    
    return points
    
def savePoints(points) :
    f = open("testFacialPoints.txt", "w")
    for p in points:
        f.write("%s %s\n" % (p[0], p[1]))
        
        
if __name__ == '__main__':
    
    # Read source images
    img1 = cv2.imread(IMAGE1_PATH)
    img2 = cv2.imread(IMAGE2_PATH)
    imgBase = cv2.imread(IMAGE_BASE_PATH)
    
    
    # Determine Facial Landmark Points
    points1 = getLandmarks(img1)
    points2 = getLandmarks(img2)
    pointsBase = getLandmarks(imgBase)
    

    # # Compute weighted average point coordinates
    alpha = 0.5
    beta = 0.3
    
    # points = []
    
    # for i in xrange(0, len(points1)):
        # x = (1 - beta) * ((1 - alpha) * points1[i][0] + alpha * points2[i][0]) + beta * pointsBase[i][0]
        # y = (1 - beta) * ((1 - alpha) * points1[i][1] + alpha * points2[i][1]) + beta * pointsBase[i][1]
        # points.append((int(x), int(y)))
        
        
    # Convert to List of tuples
    points = toListTuples(pointsBase)
    
    # Determine Delaunay Triangles of imgMorph
    triangles = getTriangles(imgBase.shape, points)
    
    # imgMorph = np.zeros(img1.shape, dtype = img1.dtype)
    imgMorph = imgBase.copy()
    
    # Morph each triangle
    for indice in triangles :
        x,y,z = indice
            
        # Rounding
        x = int(x)
        y = int(y)
        z = int(z)
        
        # Get Triangle Coordinate
        triangle1 = [points1[x], points1[y], points1[z]]
        triangle2 = [points2[x], points2[y], points2[z]]
        triangle = [ points[x], points[y], points[z] ]

        # Morph one triangle at a time.
        morphTriangle(img1, img2, imgBase, imgMorph, triangle1, triangle2, triangle, alpha, beta)

        
        
    # Seamless Cloning
    
    # Find convex hull
    hullIndex = cv2.convexHull(np.array(points), returnPoints = False)
    
    hull = []
    
    for i in xrange(0, len(hullIndex)):
        hull.append(points[hullIndex[i]])
    
    # Calculate mask
    mask = np.zeros(imgMorph.shape, dtype = imgMorph.dtype)
    
    # Calculate Rectangle
    r = cv2.boundingRect(np.float32([hull]))
    
    # Calculate center point for seamClone
    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
    
    # Clone seamlessly
    result = cv2.seamlessClone(np.uint8(imgMorph), imgBase, mask, center, cv2.NORMAL_CLONE)
    
    # Display result
    cv2.imshow("Image 1", img1)
    cv2.imshow("Image 2", img2)
    # cv2.imshow("Result", imgMorph)
    cv2.imshow("Result", result)
    cv2.waitKey()