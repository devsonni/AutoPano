#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 1 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute

Dev Soni (djsoni@wpi.edu)
Masters Student in Robotics Engineering
Worcester Polytechnic Institute
"""

# Code starts here:
import numpy as np
import cv2
import argparse
import os
from skimage.feature import peak_local_max
import random
import math
import glob


# Reading imgs form the given directory, this will be replaced in server implementation
def img_reading(folder):
    image_list = os.listdir(folder)
    image_list = [item for item in image_list if os.path.isfile(os.path.join(folder, item))]
    image_list.sort()
    imgs = []
    for file in image_list:
        file_path = os.path.join(folder, file)
        img = cv2.imread(file_path)
        y, x, c = np.shape(img)
        img = cv2.resize(img, (int(y/2), int(x/2)))
        if img is not None:
            imgs.append(img)

    n_imgs = len(image_list)
    print(f"Number of imgs Loaded: {n_imgs}")
    
    return imgs, n_imgs

# New Feature Matching -- 2nd version
def match_features_and_find_homography(InputPath, imgs, n_imgs, pipeline=True):
    """
    Match features between two images using SIFT descriptors,
    draw the matches, and find the homography matrix.
    
    :param image1: First input image.
    :param image2: Second input image.
    :return: Image containing matches and the homography matrix.
    """
    InputPath2 = InputPath
    InputPath3 = InputPath

    # This is for saving best match
    if pipeline:
        InputPath2 = str(InputPath2) + '/FMatch'
        print(InputPath2)
        if not os.path.exists(InputPath2):
            os.makedirs(InputPath2)
    
    # This is for saving feature descriptor
    if pipeline:
        InputPath3 = str(InputPath3) + '/FD'
        print(InputPath3)
        if not os.path.exists(InputPath3):
            os.makedirs(InputPath3)
    
    Hs = []
    
    for i in range(n_imgs-1):
        img = imgs[i].copy()
        img2 = imgs[i+1].copy()
        
        # Create SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors for both images
        kp1, des1 = sift.detectAndCompute(img, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        # Draw keypoints on the image
        img_key = img
        img_key_2 = img2
        img_key = cv2.drawKeypoints(img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_key_2 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Create a Brute-Force Matcher object
        bf = cv2.BFMatcher()
        
        # Match descriptors of two images
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test to get good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good_matches.append(m)
        
        # Draw matches
        img_matches = cv2.drawMatches(img, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography matrix
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 7, 0.99)
        
        Hs.append(H)
        
        # saving the features
        if pipeline:
            file_name = str(i+1) + '.png'
            output_file_path = os.path.join(InputPath3, file_name)
            cv2.imwrite(output_file_path, img_key)
            file_name = str(i+2) + '.png'
            output_file_path = os.path.join(InputPath3, file_name)
            cv2.imwrite(output_file_path, img_key_2)

        # saving the drawmatch
        if pipeline:
            file_name = str(i+1) + '.png'
            output_file_path = os.path.join(InputPath2, file_name)
            cv2.imwrite(output_file_path, img_matches)

    return Hs

# Weighted mask for blending the images
def create_weighted_mask(mask):
    # Ensure the mask is single-channel and of type uint8
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Compute distance transform
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    # Normalize the distance transform to [0, 1]
    max_dist = np.max(dist_transform)
    if max_dist > 0:
        weighted_mask = dist_transform / max_dist
    else:
        weighted_mask = dist_transform

    return weighted_mask

# wrap, stitch, blend and create a single panorama image
def Pano(imgs, n_imgs, Hs, InputPath):
    # Creating folder to save imgs
    InputPath2 = InputPath
    InputPath = str(InputPath) + '/Panorama'
    if not os.path.exists(InputPath):
        os.makedirs(InputPath)
    imgs_copy = imgs
    
    for i in range(n_imgs-1):

        inter = int(math.ceil(n_imgs/2))

        if i+1 < inter:
            
            H = np.linalg.inv(Hs[i])

             # Convert H to float32 if it's not already
            if H.dtype != np.float32 and H.dtype != np.float64:
                H = H.astype(np.float32)

            # Get the dimensions of the images
            h1, w1 = imgs[i+1].shape[:2]
            h2, w2 = imgs[i].shape[:2]

            # Get the canvas size
            corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

            # Warp the corners of img2
            corners_img2_ = cv2.perspectiveTransform(corners_img2, H)
            all_corners = np.concatenate((corners_img1, corners_img2_), axis=0)

            # Find the bounding box for the panorama
            [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

            # Translation homography
            translation_dist = [-x_min, -y_min]
            H_translation = np.array([[1, 0, translation_dist[0]],
                                      [0, 1, translation_dist[1]],
                                      [0, 0, 1]], dtype=np.float32)

            # Warp the images
            panorama_size = (x_max - x_min, y_max - y_min)
            img1_warped = cv2.warpPerspective(imgs[i+1], H_translation, panorama_size)
            img2_warped = cv2.warpPerspective(imgs[i], H_translation.dot(H), panorama_size)

            # Create a mask for blending
            mask1 = (img1_warped > 0).astype(np.uint8) * 255
            mask2 = (img2_warped > 0).astype(np.uint8) * 255
            mask = cv2.bitwise_or(mask1, mask2)

            # Combine the images
            # result = cv2.addWeighted(img1_warped, 0.5, img2_warped, 0.5, 0)

            # Create weighted masks
            weighted_mask1 = create_weighted_mask(mask1)
            weighted_mask2 = create_weighted_mask(mask2)

            # Combine the images using the weights
            result = (img1_warped * weighted_mask1[..., None] + img2_warped * weighted_mask2[..., None]) /\
                (weighted_mask1[..., None] + weighted_mask2[..., None])

            # saving the drawmatch
            file_name = str(i) + '.png'
            output_file_path = os.path.join(InputPath, file_name)
            cv2.imwrite(output_file_path, result)

        else:
            H = Hs[i]

             # Convert H to float32 if it's not already
            if H.dtype != np.float32 and H.dtype != np.float64:
                H = H.astype(np.float32)

            # Get the dimensions of the images
            h1, w1 = imgs[i].shape[:2]
            h2, w2 = imgs[i+1].shape[:2]

            # Get the canvas size
            corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

            # Warp the corners of img2
            corners_img2_ = cv2.perspectiveTransform(corners_img2, H)
            all_corners = np.concatenate((corners_img1, corners_img2_), axis=0)

            # Find the bounding box for the panorama
            [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

            # Translation homography
            translation_dist = [-x_min, -y_min]
            H_translation = np.array([[1, 0, translation_dist[0]],
                                      [0, 1, translation_dist[1]],
                                      [0, 0, 1]], dtype=np.float32)

            # Warp the images
            panorama_size = (x_max - x_min, y_max - y_min)
            img1_warped = cv2.warpPerspective(imgs[i], H_translation, panorama_size)
            img2_warped = cv2.warpPerspective(imgs[i+1], H_translation.dot(H), panorama_size)

            # Create a mask for blending
            mask1 = (img1_warped > 0).astype(np.uint8) * 255
            mask2 = (img2_warped > 0).astype(np.uint8) * 255
            mask = cv2.bitwise_or(mask1, mask2)

            # Combine the images
            # result = cv2.addWeighted(img1_warped, 0.5, img2_warped, 0.5, 0)
            # Create weighted masks
            weighted_mask1 = create_weighted_mask(mask1)
            weighted_mask2 = create_weighted_mask(mask2)

            # Combine the images using the weights
            result = (img1_warped * weighted_mask1[..., None] + img2_warped * weighted_mask2[..., None]) /\
                (weighted_mask1[..., None] + weighted_mask2[..., None])

            # saving the drawmatch
            file_name = str(i) + '.png'
            output_file_path = os.path.join(InputPath, file_name)
            cv2.imwrite(output_file_path, result)

    # reading the intermediate panorama -- add it back after blending is fixed
    folder = "../Data/Train/Set" + str(InputPath2[-1]) + "/Panorama"
    image_list = os.listdir(folder)
    image_list = [item for item in image_list if os.path.isfile(os.path.join(folder, item))]
    print("Stitching: ", image_list)
    image_list.sort()
    intermediate = []
    for file in image_list:
        file_path = os.path.join(folder, file)
        img = cv2.imread(file_path)
        if img is not None:
            intermediate.append(img)

    image_files = glob.glob(os.path.join(folder, '*'))
    # Delete only files, not directories
    if not len(intermediate) == 1:
        [os.remove(f) for f in image_files if os.path.isfile(f)]

    # print("Stitching: " + str(os.listdir(folder)))
    for i in range(len(intermediate)-1):
        Pipeline(InputPath2, intermediate, len(intermediate), pipeline=False)


def Pipeline(InputPath, imgs, n_imgs, pipeline=True):

    # gettign homography
    H = match_features_and_find_homography(InputPath, imgs, n_imgs, pipeline)

    # Panorama
    Pano(imgs, n_imgs, H, InputPath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--NumFeatures', default=100, help="Number of best features to extract from each image, Default:100")
    parser.add_argument('--Path', default="../Data/Train/Set1", help="Enter the folder name to imgs from your directory")
    Args = parser.parse_args()
    InputPath = Args.Path

    # Loading the imgs
    imgs, n_imgs = img_reading(InputPath)

    Pipeline(InputPath, imgs, n_imgs)

if __name__ == "__main__":
    main()
