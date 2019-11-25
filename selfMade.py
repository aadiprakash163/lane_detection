
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os


def read_image(image_path):
    return mpimg.imread(image_path)
#test_image = read_image('test.jpg')

import math

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    
    ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
 
    return cv2.addWeighted(initial_img, α, img, β, λ)

def extrapolate_lines(lines,original_image):
    pos_slope_lines = []
    neg_slope_lines = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if (y2-y1)/(x1-x2) > 0:
                pos_slope_lines.append([x1,y1,x2,y2])
            else:
                neg_slope_lines.append([x1,y1,x2,y2])

    pos_line_par = []
    neg_line_par = []
    #print(pos_slope_lines,"\n\n",neg_slope_lines)
    for i in range(len(pos_slope_lines)):
            x1 = pos_slope_lines[i][0]
            y1 = pos_slope_lines[i][1]
            x2 = pos_slope_lines[i][2]
            y2 = pos_slope_lines[i][3]
            slope = (y2-y1)/(x1-x2)
            intercept = -1*y1 - slope*x1
            pos_line_par.append([slope,intercept])

    for i in range(len(neg_slope_lines)):
            x1 = neg_slope_lines[i][0]
            y1 = neg_slope_lines[i][1]
            x2 = neg_slope_lines[i][2]
            y2 = neg_slope_lines[i][3]
            slope = (y2-y1)/(x1-x2)
            intercept = -1*y1 - slope*x1
            neg_line_par.append([slope,intercept])

    pos_slopes = [pos_line_par[i][0] for i in range(len(pos_line_par)) ]
    pos_intercepts = [pos_line_par[i][1] for i in range(len(pos_line_par)) ]
    neg_slopes = [neg_line_par[i][0] for i in range(len(neg_line_par)) ]
    neg_intercepts = [neg_line_par[i][1] for i in range(len(neg_line_par)) ]

    mean_pos_slope = np.mean(pos_slopes)
    mean_pos_intercept = np.mean(pos_intercepts)
    mean_neg_slope = np.mean(neg_slopes)
    mean_neg_intercept = np.mean(neg_intercepts)


    intercestion_point_x = int((mean_neg_intercept - mean_pos_intercept)/(mean_pos_slope - mean_neg_slope))
    intercestion_point_y = int(-mean_pos_slope*intercestion_point_x - mean_pos_intercept)


    line_image = np.zeros_like(original_image)
    imshape = line_image.shape
    pos_x1 = int((- mean_pos_intercept-imshape[0])/mean_pos_slope)
    neg_x1 = int((- mean_neg_intercept-imshape[0])/mean_neg_slope)

    print(pos_x1,imshape[0],neg_x1)

    extrapolated_lines_image = cv2.line(line_image,(pos_x1,imshape[0]),(intercestion_point_x,intercestion_point_y),[255,0,0],10)
    extrapolated_lines_image = cv2.line(extrapolated_lines_image,(neg_x1,imshape[0]),(intercestion_point_x,intercestion_point_y),[255,0,0],10)
    vertices = np.array([[(150,550), (400,350), (600,350), (950,550)]], dtype=np.int32)
    extrapolated_masked_image = region_of_interest(extrapolated_lines_image, vertices)
    return extrapolated_masked_image


    
def draw_lines(image):
    
    gray_image = grayscale(image)
    
    kernel_size = 5
    blurrFree_image = gaussian_blur(gray_image,kernel_size)

    low_threshold = 50
    high_threshold = 150
    canny_image = canny(blurrFree_image,low_threshold,high_threshold)
    plt.subplot(2,2,1)
    plt.imshow(canny_image, cmap='gray')
    
    vertices = np.array([[(150,550), (400,350), (600,350), (950,550)]], dtype=np.int32)
    masked_image = region_of_interest(canny_image, vertices)
    plt.subplot(2,2,2)
    plt.imshow(masked_image, cmap='gray')

    rho = 2
    theta = np.pi/180
    threshold = 15
    min_line_len = 40
    max_line_gap = 100
    lines = hough_lines(masked_image,rho,theta,threshold,min_line_len,max_line_gap)

    #for line in lines:
     #   for x1,y1,x2,y2 in lines[]:
      #      print(x1,y1,x2,y2)
    extended_lines_image = extrapolate_lines(lines,image)
    plt.subplot(2,2,3)
    plt.imshow(extended_lines_image)
    
    final_image = weighted_img(extended_lines_image,image)
    return(final_image)
    plt.subplot(2,2,4)
    plt.imshow(final_image)

def process_image(image):
    """Puts image through pipeline and returns 3-channel image for processing video below."""
    result = draw_lines(image)
    return result

from moviepy.editor import VideoFileClip
Output_vid1 = 'Op_vid1.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
V_clip = clip1.fl_image(process_image)
V_clip.write_videofile(Output_vid1, audio=False)
#draw_lines(test_image)
#plt.show()