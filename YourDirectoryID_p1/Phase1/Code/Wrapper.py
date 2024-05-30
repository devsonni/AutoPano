#!/usr/bin/env python

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
from skimage.feature import peak_local_max

# Reading imgs form the given directory, this will be replaced in server implementation
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
        threshold = 0.001 * dst.max()
        
        # Discarding weak points
        dst[dst <= threshold] = 0
        c_scores_list.append(dst)

        # Extracting the coordinates of the corners
        corner_coords = np.argwhere(dst > 0)
        corners = [(int(x), int(y)) for y, x in corner_coords]
        num_corners = len(corners)
        corner_coords_list.append(corners)
        num_corners_list.append(num_corners)

        # Creating a copy of the image
        img_copy = img.copy()
        # Applying red points to the detected corners
        img_copy[dst > threshold] = [0, 0, 255]

        # Saving the imgs
        file_name = str(i+1) + '.png'
        output_file_path = os.path.join(InputPath, file_name)
        cv2.imwrite(output_file_path, img_copy)

    return corner_coords_list, num_corners_list, c_scores_list


# ANMS: Adaptive Non-Maximal Suppression
def ANMS(imgs, n_imgs, corner_coords_list, num_corners_list, c_score_list, InputPath, N_best=200):

    InputPath = str(InputPath) + '/ANMS'
    if not os.path.exists(InputPath):
        os.makedirs(InputPath)
    
	# Return
    selected_point_list = []

    # Iterating through each imgs
    for ig in range(n_imgs):
        # computing the regional peak maxima 
        coords = peak_local_max(c_score_list[ig], min_distance=2, threshold_abs=0.01)
        r = np.full(len(coords)-1, np.inf)
        for i in range(len(coords)-1):
            for j in range(len(coords)-1):
                if c_score_list[ig][coords[i][0], coords[i][1]] < c_score_list[ig][coords[j][0], coords[j][1]]:
                    ED = np.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
                    if ED < r[i]:
                         r[i] = ED
        
        # Sort the distances in descending order
        sorted_indices = np.argsort(r)[::-1]
        N_best = min(N_best, len(coords))
        selected_indices = sorted_indices[:N_best]

        # Extracting coordinates of the selected points
        selected_pts = np.zeros((N_best, 2))
        for i in range(N_best-1):
            selected_pts[i] = coords[selected_indices[i]]
        selected_pts = selected_pts.astype(int)

        selected_point_list.append(selected_pts)

        # Creating a copy of the image
        img_copy = imgs[ig].copy()
        
        # plotting the points on img
        for i in range(N_best):
             cv2.circle(img_copy, (selected_pts[i][1], selected_pts[i][0]), 2, (0, 0, 255), -1)

        # Saving the imgs
        file_name = str(ig+1) + '.png'
        output_file_path = os.path.join(InputPath, file_name)
        cv2.imwrite(output_file_path, img_copy)

    return selected_point_list

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
    ANMS(imgs, n_imgs, corner_coords_list, num_corners_list, c_score_list, InputPath)

if __name__ == "__main__":
    main()
