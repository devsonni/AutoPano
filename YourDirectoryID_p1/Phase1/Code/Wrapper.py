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
        if img is not None:
            imgs.append(img)

    n_imgs = len(image_list)
    print(f"Number of imgs Loaded: {n_imgs}")
    
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
def ANMS(imgs, n_imgs, corner_coords_list, num_corners_list, c_score_list, InputPath, N_best=1000):

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
def FD(imgs, n_imgs, n_best_pts):
    FDs= []
    F_coords = []
    # Going through each imgs
    for ig in range(n_imgs):
        img = imgs[ig].copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.float32(img)
        FDi = []
        Coords = []
        # creating path of 41x41 to the main img
        for j in range(len(n_best_pts[ig])):
            x1, y1 = n_best_pts[ig][j][0]-20, n_best_pts[ig][j][1]-20
            x2, y2 = n_best_pts[ig][j][0]+21, n_best_pts[ig][j][1]+21
            s_y, s_x = np.shape(img)
            if x1 > 0 and y1 > 0 and x2 < s_x and y2 < s_y:
                patch_center = [n_best_pts[ig][j][0], n_best_pts[ig][j][1]]
                patch = img[x1:x2, y1:y2]
                patch = cv2.GaussianBlur(patch,(7,7),cv2.BORDER_DEFAULT)
                patch = cv2.resize(patch, (8,8))
                patch = cv2.resize(patch, (64,1))
                feature_vector = abs(np.mean(patch) - patch)/np.std(patch)
                FDi.append(feature_vector)
                Coords.append(patch_center)
        FDs.append(FDi)
        F_coords.append(Coords)
    return FDs, F_coords

# Feature Descriptors (FDs) Using of SIFT
def Features_(InputPath, imgs, n_imgs):
    # Creating folder to save imgs
    InputPath = str(InputPath) + '/SIFT'
    if not os.path.exists(InputPath):
        os.makedirs(InputPath)

    FDs = []
    for i in range(n_imgs):
        img = imgs[i].copy()
        img2 = imgs[i].copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.float32(img)

        sift = cv2.SIFT_create()

        # Convert corners to KeyPoint objects
        # keypoints = [cv2.KeyPoint(x=float(x), y=float(y), size=1) for (x, y) in n_best_pts[i]]

        # Compute the SIFT descriptors at the given keypoints
        keypoints, descriptors = sift.detectAndCompute(imgs[i], None)
        # Convert keypoints to (x, y) coordinates
        keypoints_coords = np.array([kp.pt for kp in keypoints])

        FDs.append(descriptors)

        for j in range(len(keypoints_coords)):
            cv2.circle(img2, (int(keypoints_coords[j][0]), int(keypoints_coords[j][1])), 2, (0, 0, 255), -1)
        
        # saving the drawmatch
        file_name = str(i+1) + '.png'
        output_file_path = os.path.join(InputPath, file_name)
        cv2.imwrite(output_file_path, img2)
    
    return FDs, 

# New Feature Matching -- 2nd version
def match_features_and_find_homography(InputPath, imgs, n_imgs, pipeline):
    """
    Match features between two images using SIFT descriptors,
    draw the matches, and find the homography matrix.
    
    :param image1: First input image.
    :param image2: Second input image.
    :return: Image containing matches and the homography matrix.
    """
    InputPath2 = InputPath
    # Creating folder to save imgs
    if not pipeline:
        InputPath2 = str(InputPath2) + '/FMatch'
        if not os.path.exists(InputPath2):
            os.makedirs(InputPath2)
    
    Hs = []
    
    for i in range(n_imgs-1):
        img = imgs[i].copy()
        img2 = imgs[i+1].copy()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = np.float32(img)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # img2 = np.float32(img2)
        
        # Create SIFT detector
        sift = cv2.SIFT_create()
        
        print(img.shape)
        print(img2.shape)

        # Detect keypoints and compute descriptors for both images
        kp1, des1 = sift.detectAndCompute(img, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
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

        # saving the drawmatch
        if not pipeline:
            file_name = str(i+1) + '.png'
            output_file_path = os.path.join(InputPath2, file_name)
            cv2.imwrite(output_file_path, img_matches)

    return Hs

# Feature Matching
def FM(InputPath, imgs, n_imgs, Features, F_Coords):
    # Creating folder to save imgs
    InputPath = str(InputPath) + '/FM'
    if not os.path.exists(InputPath):
        os.makedirs(InputPath)
    
    # Matching features to each pair of the imgs -> loop - pairs, features1, all features2 
    feature_match = []
    for ig in range(n_imgs-1):
        f_match = []
        for i in range(len(Features[ig])):
            sq_dist = []
            for j in range(len(Features[ig+1])):
                dist = np.sum(np.sqrt((Features[ig][i] - Features[ig+1][j])**2))
                sq_dist.append(dist)
            
            sorted_indices = np.argsort(sq_dist)
            if len(sorted_indices) > 1 and np.isfinite(sq_dist[sorted_indices[1]]) and sq_dist[sorted_indices[1]] != 0:
                if sq_dist[sorted_indices[0]] / sq_dist[sorted_indices[1]] < 0.7:
                    f_match.append((F_Coords[ig][i], F_Coords[ig+1][sorted_indices[0]]))
        
        # Appending match point coordinates to the list
        feature_match.append(f_match)

    # Drawing the image match
    for i in range(len(feature_match)):
        # merging two imgs
        img_copy = np.concatenate((imgs[i], imgs[i+1]), axis=1)

        # Writing custom drawing
        for (x1, y1), (x2, y2) in feature_match[i]:
            cv2.circle(img_copy, (int(y1), int(x1)), 1, (0, 0, 255), 1)  # Red border circle
            cv2.circle(img_copy, (int(y2 + imgs[i].shape[1]), int(x2)), 1, (0, 0, 255), 1)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random color
            cv2.line(img_copy, (y1, x1), (y2 + imgs[i].shape[1], x2), color, 1)

        # saving the drawmatch
        file_name = str(i+1) + '.png'
        output_file_path = os.path.join(InputPath, file_name)
        cv2.imwrite(output_file_path, img_copy)

    return feature_match


# Estimate Homography, and RANSAC
def RanHomo(InputPath, imgs, n_imgs, Matching, reprojection_threshold=2, max_iters=5000, confidence=0.8):
    # Creating folder to save imgs
    InputPath = str(InputPath) + '/RANSAC'
    if not os.path.exists(InputPath):
        os.makedirs(InputPath)
    
    Hs = []
    Inliers_ = []
    masks = []
    # looping through each pairs
    for ig in range(n_imgs-1):
        src_pts = np.float32([match[0] for match in Matching[ig]]).reshape(-1, 1, 2)
        dst_pts = np.float32([match[1] for match in Matching[ig]]).reshape(-1, 1, 2)
    
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reprojection_threshold, maxIters=max_iters, confidence=confidence)
    
        inliers = mask.ravel().tolist()

        Hs.append(H)
        Inliers_.append(inliers)
        masks.append(mask)

    filtered_matches_ = []
    for ig in range(n_imgs-1):
        filtered_matches = [match for match, inlier in zip(Matching[ig], inliers) if inlier]
        filtered_matches_.append(filtered_matches)
    
    # Drawing the matches on the imgs
    for i in range(len(filtered_matches_)):
        # merging two imgs
        img_copy = np.concatenate((imgs[i], imgs[i+1]), axis=1)

        # Writing custom drawing
        for (x1, y1), (x2, y2) in filtered_matches_[i]:
            cv2.circle(img_copy, (y1, x1), 1, (0, 0, 255), 1)  # Red border circle
            cv2.circle(img_copy, (y2 + imgs[i].shape[1], x2), 1, (0, 0, 255), 1)

            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random color
            cv2.line(img_copy, (y1, x1), (y2 + imgs[i].shape[1], x2), color, 1)

        # saving the drawmatch
        file_name = str(i+1) + '.png'
        output_file_path = os.path.join(InputPath, file_name)
        cv2.imwrite(output_file_path, img_copy)

    return Hs

# Wrapping and Pano
# trying for only two images, for the first case only   
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
            result = cv2.addWeighted(img1_warped, 0.5, img2_warped, 0.5, 0)

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
            result = cv2.addWeighted(img1_warped, 0.5, img2_warped, 0.5, 0)

            # saving the drawmatch
            file_name = str(i) + '.png'
            output_file_path = os.path.join(InputPath, file_name)
            cv2.imwrite(output_file_path, result)

    # reading the intermediate panorama
    folder = "../Data/Train/Set" + str(InputPath2[-1]) + "/Panorama"
    image_list = os.listdir(folder)
    image_list = [item for item in image_list if os.path.isfile(os.path.join(folder, item))]
    print(image_list)
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
        
    print("after delete:" + str(os.listdir(folder)))
    for i in range(len(intermediate)-1):
        Pipeline(InputPath2, intermediate, len(intermediate))

        

def Blending():
    pass

def Pipeline(InputPath, imgs, n_imgs):

    H = match_features_and_find_homography(InputPath, imgs, n_imgs, pipeline=1)
    # Feature Matching
    # Matching = FM(InputPath, imgs, n_imgs, Features, n_best_pts)

    # RANSAC and Estimating Homography
    # H = RanHomo(InputPath, imgs, n_imgs, Matching)

    # Panorama
    Pano(imgs, n_imgs, H, InputPath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--NumFeatures', default=100, help="Number of best features to extract from each image, Default:100")
    parser.add_argument('--Path', default="../Data/Train/Set1", help="Enter the folder name to imgs from your directory")
    Args = parser.parse_args()
    InputPath = Args.Path
    NumFeatures = Args.NumFeatures

    # Loading the imgs
    imgs, n_imgs = img_reading(InputPath)
    
    # Corner detection
    # corner_coords_list, num_corners_list, c_score_list = corner_detection(InputPath, imgs, n_imgs)
    
    # ANMS
    # n_best_pts = ANMS(imgs, n_imgs, corner_coords_list, num_corners_list, c_score_list, InputPath)

    # Feature Descriptors
    # Features = Features_(InputPath, imgs, n_imgs)

    # H = match_features_and_find_homography(InputPath, imgs, n_imgs)
    # # Feature Matching
    # # Matching = FM(InputPath, imgs, n_imgs, Features, n_best_pts)

    # # RANSAC and Estimating Homography
    # # H = RanHomo(InputPath, imgs, n_imgs, Matching)

    # # Panorama
    # Pano(imgs, n_imgs, H, InputPath)

    Pipeline(InputPath, imgs, n_imgs)

if __name__ == "__main__":
    main()
