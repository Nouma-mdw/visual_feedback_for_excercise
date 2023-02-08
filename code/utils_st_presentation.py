import os
import shutil
import cv2
import numpy as np
import time
import pickle

import io
# https://www.digitalocean.com/community/tutorials/python-io-bytesio-stringio

class new_exercise:

	def __init__(self, exercise_name, video_path):
		self.path = video_path
		self.exercise = exercise_name
		self.src_folder = 'best_forms_collection'
		self.new_path = self.src_folder + '/' + self.exercise

	def create_path(self):
		shutil.rmtree(self.new_path, ignore_errors=True)
	
		# Now, create new empty train and test folders
		os.makedirs(self.new_path)


	def load_file(self):
		# shutil.copy2(src, self.new_path/<file>..mp4
		shutil.copy2(self.path, self.new_path + '/' + self.exercise + '.mp4')
        
        
	def pickle_save(self, file, name = "str"):
		with open(self.new_path + '/' + self.exercise + '__' + name  , 'wb') as f:
			pickle.dump(file, f)
            
            
            
def transform_keypoints(one_frame_dot_landmark):
    reference_img = []
    for idx, data_point in enumerate(one_frame_dot_landmark.landmark): #################  ACCESS KEYPOINTS FOR COSINE SIMILARITY
        reference_img.extend([data_point.x,
                                   data_point.y, 
                                   data_point.z,
                                 # data_point.visibility
                                 ])
    return [reference_img]
            
	

        
def extract_keypoints(image_path, Frame = False, save = False):
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    if Frame == False:
        # instance of mp_pose with selected parameters


        # cv2.imread() reads an image.
        img = cv2.imread(image_path, -1)

            # 
        # extracting landmarks with instance
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # saving coordinates of every keypoint in a list
        keypoints = results.pose_landmarks

        # save file with pickle
        if save == True:
            im_name = image_path.split("\\")[-2]
            with open(im_name, 'wb') as f:
                pickle.dump(keypoints, f)

            with open(im_name, 'rb') as f:
                x = pickle.load(f)
            return x        
        return keypoints
    else:
        results = pose.process(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
        keypoints = results.pose_landmarks
        return keypoints
    
def extract_vid_keypoints(vid_path, Frame = False, save = False):
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    if Frame == False:
        # instance of mp_pose with selected parameters
        
        
        cv2.waitKey(0)


        # cv2.imread() reads an image.
        img = cv2.imshow(image_path, -1)

            # 
        # extracting landmarks with instance
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # saving coordinates of every keypoint in a list
        keypoints = results.pose_landmarks

        # save file with pickle
        if save == True:
            im_name = image_path.split("\\")[-2]
            with open(im_name, 'wb') as f:
                pickle.dump(keypoints, f)

            with open(im_name, 'rb') as f:
                x = pickle.load(f)
            return x        
        return keypoints
    else:
        results = pose.process(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
        keypoints = results.pose_landmarks
        return keypoints
    