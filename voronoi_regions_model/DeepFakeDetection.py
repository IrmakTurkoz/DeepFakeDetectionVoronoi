import numpy as np
import os
import cv2
import sys
import time
import dlib
import glob
import argparse
import voronoi as v

def checkDeepFake(regions):
    return True

    
def initialize_predictor():
    # Predictor
    ap = argparse.ArgumentParser()
    if len(sys.argv) > 1:
        predictor_path = sys.argv[1]

        print("[INFO] loading facial landmark predictor...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        return predictor,detector
    else:
        print("ERROR : Please give the model as argument.")
        return None,None 

def extract_features(fileDirectory,videos,labels,show_results = False,frame_rate = 50):
    predictor,detector = initialize_predictor()
    if predictor is None:
        return
    for filename in videos:
        currentfile = os.path.join(fileDirectory,filename)
        if currentfile:
            print('Opening the file with name ' + currentfile)  
            cap = cv2.VideoCapture(currentfile)
            face_id = 0
            while(cap.isOpened() and not(cv2.waitKey(1) & 0xFF == ord('q'))):
                prev_features = []
                ret, frame = cap.read()
                features = []
                if frame is None:
                    break 
                img = v.preprocessing(frame)
                regions = detector(img, 0)
                if regions:
                    # loop over the face detections
                    for (i, rect) in enumerate(regions):
                        # determine the facial landmarks for the face region, then
                        # convert the facial landmark (x, y)-coordinates to a NumPy
                        # array
                        vor_features = v.createVoronoi(img,predictor,rect,face_id + i,show_results=show_results)

                        features.append(vor_features)
                        if show_results and cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    face_id =+ 1
                if show_results:
                    cv2.imshow("Frame",img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if show_results:
                cap.release()
                cv2.destroyAllWindows()
        else: 
            print("Could not find the directory")

def pad_images(fileDirectory):
    max_w = 80
    max_h = 100
    # for filename in os.listdir(fileDirectory):
    #     currentfile = os.path.join(fileDirectory,filename)
    #     if currentfile:
    #         img = cv2.imread(currentfile)

    #         ht, wd, cc = img.shape
    #         if ht > max_h:
    #             max_h = ht
    #         if wd > max_w: 
    #             max_w = wd


    print("Max_w #{} Max_h #{}",max_w, max_h)
    for filename in os.listdir(fileDirectory):
        currentfile = os.path.join(fileDirectory,filename)
        if currentfile:
            img = cv2.imread(currentfile) 
            ht, wd, cc= img.shape
            result = np.full((max_h,max_w,cc), (0,0,0), dtype=np.uint8)

            # compute center offset
            xx = (max_w - wd) // 2
            yy = (max_h - ht) // 2

            # copy img image into center of result image
            result[yy:yy+ht, xx:xx+wd] = img
            cv2.imwrite("features/"+ filename, result)