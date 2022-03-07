import numpy as np
import cv2
from scipy.spatial import Voronoi
from scipy.spatial import Delaunay
from imutils import face_utils
import math
import time
import imutils
import dlib
from PIL import Image, ImageDraw
COUNT = 0

def increment():
    global COUNT
    COUNT = COUNT+1


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

#Define preprocess here
def preprocessing(rgbimg):
    rgbimg = imutils.resize(rgbimg, height= 1200,width=1600)
    labimg = cv2.cvtColor(rgbimg, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(labimg)    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final

def crop_image(polygon,destination, img):

    ## (1) Crop the bounding rect
    pts = np.asarray(polygon, dtype= np.int32)

    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    black = 0
    # print(dst.shape)
    for i in dst:
        for j in i:
            if all(j == 0):
                black += 1

    if black/ (dst.shape[0]*dst.shape[1] )<0.4  and dst.shape[0] > 20 and dst.shape[1] > 20: 
        # im = Image.fromarray(dst,mode='RGB')
        cv2.imwrite(destination + str(COUNT) + ".jpg", dst)
        increment()
    return dst

def createVoronoi(img,destination,predictor,rect,face_id,show_results = False):
    
    # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.putText(img, "Face #{}".format(face_id + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 150, 220), 1)
    x = max(0, min(x, img.shape[1]-1))
    y = max(0, min(y, img.shape[0]-1))
    black = np.zeros((h, w, 1), dtype = "uint8")
    cutted_img = img[y:y+h,x:x+w,:]
    cutted_img = adjust_gamma(cutted_img)
    # orginal_cutted_img = cutted_img
    # shape = predictor(cutted_img, dlib.rectangle(0,0,w,h))
    # landmarks = face_utils.shape_to_np(shape)

    # cv2.rectangle(img, (x, y), (x + w, y + h), (150, 60, 60), 2)         
    # vor = Voronoi(landmarks)
    
    # delu = Delaunay(landmarks)
    # vor_vertices =  vor.vertices
    # vertices_in_region = []

    # masks = []
    # csv_file = open('features/features.csv','w') 
                  
    # for convex_vertices in vor.regions:
    #     if all(v > 0 for v in convex_vertices):
    #         polygon = []
    #         for indx in convex_vertices:
    #             if(vor_vertices[indx][0] > 0 and vor_vertices[indx][1]>0):
    #                 polygon.append(vor_vertices[indx])
    #         if polygon:
    #             mask = crop_image(polygon,destination,orginal_cutted_img)
    #             masks.append(mask)
    #             ##text=List of strings to be written to file
    #             # csv_file.write(str(face_id))
    #             if show_results:
    #                 cv2.imshow("cropped",mask)
            
    #         break
    # # print(vor.regions)
    # if show_results:
    #     cv2.imshow("Face features" + str(face_id),cutted_img)
            
        # for edge in delu.simplices:
        #     print(edge)
        # for (x, y) in vor.vertices:
        #     cv2.circle(img, (math.floor(x), math.floor(y)), 1, (190, 220, 255), -1)





    
    
    # for vpair in vor.ridge_vertices:
    #     if vpair[0] >= 0 and vpair[1] >= 0:
    #         v0 = vor_vertices[vpair[0]]
    #         v1 = vor_vertices[vpair[1]]
    #         # if v0[0] > 0 and v1[0] > 0 and v0[1] > 0 and v1[1] > 0 and v0[0] < h and v0[1] < h and v1[0] < w and v1[1] < w: 
    #         # if not(x < v0[0] and v0[0] < x + w and x < v1[0] and v1[0] < x + w and y < v0[1] and v0[1] < y + h and y < v1[1] and v1[1] < y + h):
    #         #     print(v0,v1)
    #         # Draw a line from v0 to v1.image = cv2.line(image, start_point, end_point, color, thickness) 
    #         vertices_in_region.append(((v0[0],v0[1]),(v1[0],v1[1])))


    #         if show_results:
    #             cv2.line(cutted_img,(math.floor(v0[0]),math.floor(v0[1])), (math.floor(v1[0]), math.floor(v1[1])),(160, 25, 190) ,2)
    #             cv2.circle(cutted_img, (math.floor(v0[0]), math.floor(v0[1])), 1, (190, 220, 255), -1)
    #             cv2.circle(cutted_img, (math.floor(v1[0]), math.floor(v1[1])), 1, (190, 220, 255), -1)
    #             # print(vor.regions[vpair[0]],vor.regions[vpair[1]])
    # return vertices_in_region


    cv2.imwrite(destination +"/" +str(COUNT) + ".jpg", cutted_img)
    increment()
    return cutted_img



