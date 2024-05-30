#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 1 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:
import numpy as np
import cv2
import argparse
import os

# Reading imgs form the giver directory, this will be replaced in server implementation
def img_reading(folder):
	image_list = os.listdir(folder)
	image_list = [item for item in image_list if os.path.isfile(os.path.join(folder, item))]
	image_list.sort()
	imgs = []
	for file in os.listdir(folder):
		file_path = os.path.join(folder, file)
		img = cv2.imread(file_path)
		if img is not None:
			imgs.append(img)

	n_imgs = len(image_list)
	print(f"Number of Images Loaded: {n_imgs}")
	
	return imgs, n_imgs  

# Corner Detection, Save Corner detection output as corners.png
def corner_detection(InputPath, imgs, n_imgs):
	InputPath = str(InputPath) + '/CornerDetection'
	if not os.path.exists(InputPath):
		os.makedirs(InputPath)		
	corner_coords_list = []
	num_corners_list = []
	c_scores_list = []

	for i in range(n_imgs):
		img = imgs[i]
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
		gray = np.float32(gray)
		dst = cv2.cornerHarris(gray,3,5,0.1)

		# Result is dilated for marking the corners, not important
		dst = cv2.dilate(dst,None)

		# Extracting the coordinates of corner points in img
		threshold = 0.001*dst.max()
		
		# Discarding weak points
		dst[dst<= threshold] = 0
		c_scores_list.append(dst)

		# Extracting the coordinates of the corners
		corner_coords = np.argwhere(dst > 0)
		corners = [(int(x), int(y)) for y, x in corner_coords]
		num_corners = len(corners)
		corner_coords_list.append(corners)
		num_corners_list.append(num_corners)

		# Applying red points of the detected corners
		img[dst>threshold]=[0,0,255]

		# Saving the imgs
		file_name = str(i+1) + '.png'
		output_file_path = os.path.join(InputPath, file_name)
		cv2.imwrite(output_file_path, img)

	return corner_coords_list, num_corners_list, c_scores_list


# ANMS: Adaptive Non-Maximal Supression
def ANMS(imgs, n_imgs, corner_coords_list, num_corners_list, c_score_list, N_best=50):
	# Initializing r-vector containing inf values
	r = np.full(N_best, np.inf)

	
	

# Feature Descriptors (FDs)
def FD():
	pass

# Feature Matching
def FM():
	pass

# Estimate Homography, and RANSAC
def RANHomo():
	pass

# Wrapping and Blending
def Blending():
	pass


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--NumFeatures', default=100, help="Number of best features to extract from each image, Default:100")

	parser.add_argument('--Path', default="../Data/Train/Set1", help="Enter the folder name to images from your directory")
	Args = parser.parse_args()
	InputPath = Args.Path
	NumFeatures = Args.NumFeatures

	# Loading the images
	imgs, n_imgs = img_reading(InputPath)
	
	# Corner detection
	corner_coords_list, num_corners_list, c_score_list = corner_detection(InputPath, imgs, n_imgs)
	
	# ANMS
	ANMS(imgs, n_imgs, corner_coords_list, num_corners_list, c_score_list)

if __name__ == "__main__":
    main()
