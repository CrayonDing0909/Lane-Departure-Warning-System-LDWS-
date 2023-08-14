import numpy as np
import cv2
import math
from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw, ImageFont

last_left_line = [100000,100000,100000,100000]
last_right_line = [100000,100000,100000,100000]

def process_image(img):
    check = 0
    gray_image = grayscale(img)
    gaus_blur = gaussian_blur(gray_image, 3)
    edges = canny(gaus_blur, 50,100)
    imshape = img.shape
    
    vertices = np.array([[(0,imshape[0]),(imshape[1]/2,500), (imshape[1]/2, 500), (imshape[1],imshape[0])]], dtype=np.int32)    
    masked = region_of_interest(edges, vertices)
    
    rho = 1       		#半徑的分辨率
    theta = np.pi/180 	#角度分辨率
    threshold = 20   	#判斷直線點數的臨界值
    min_line_len = 5    #線段長度臨界值
    max_line_gap = 10  	#線段上最近兩點之間的臨界值
    line_image, check = hough_lines(masked, rho, theta, threshold, min_line_len, max_line_gap)

    if(check):
    	img = draw_text(img)

    result = weighted_img(line_image, img)
    return result

def grayscale(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
	return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
	mask = np.zeros_like(img)
	if len(img.shape) > 2:
		channel_count = img.shape[2]
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
	global lastLineL ,lastLineR
	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

	line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
	check = draw_lines(line_img, lines)

	return line_img, check

def draw_lines(img, lines, thickness=5):
	global last_left_line, last_right_line
	top = 300
	bottom = 720
	repair = 0
	left_x1_set = []
	left_y1_set = []
	left_x2_set = []
	left_y2_set = []
	right_x1_set = []
	right_y1_set = []
	right_x2_set = []
	right_y2_set = []

	for line in lines:
		for x1,y1,x2,y2 in line:
			cv2.line(img, (x1, y1), (x2, y2), [0,255,255], thickness)
			slope = get_slope(x1,y1,x2,y2)
			if slope < 0:
				if slope > -0.5 or slope < -0.8:
					continue        
				left_x1_set.append(x1)
				left_y1_set.append(y1)
				left_x2_set.append(x2)
				left_y2_set.append(y2)
			else:
				if slope < 0.5 or slope > 0.8:
					continue        
				right_x1_set.append(x1)
				right_y1_set.append(y1)
				right_x2_set.append(x2)
				right_y2_set.append(y2)
	try:
		avg_right_x1 = int(np.mean(right_x1_set))
		avg_right_y1 = int(np.mean(right_y1_set))
		avg_right_x2 = int(np.mean(right_x2_set))
		avg_right_y2 = int(np.mean(right_y2_set))
		right_slope = get_slope(avg_right_x1,avg_right_y1,avg_right_x2,avg_right_y2)

		right_y1 = top
		right_x1 = int(avg_right_x1 + (right_y1 - avg_right_y1) / right_slope)
		right_y2 = bottom
		right_x2 = int(avg_right_x1 + (right_y2 - avg_right_y1) / right_slope)
		right_line = [right_x1, right_y1, right_x2, right_y2]

		
		last_right_line = right_line
		xR = get_cross_point(right_line[0], right_line[1] ,right_line[2], right_line[3])
	
	except ValueError:
		repair = 1
		right_line = last_right_line
		xR = get_cross_point(right_line[0], right_line[1] ,right_line[2], right_line[3])

	try:
		avg_left_x1 = int(np.mean(left_x1_set))
		avg_left_y1 = int(np.mean(left_y1_set))
		avg_left_x2 = int(np.mean(left_x2_set))
		avg_left_y2 = int(np.mean(left_y2_set))
		left_slope = get_slope(avg_left_x1,avg_left_y1,avg_left_x2,avg_left_y2)

		left_y1 = top
		left_x1 = int(avg_left_x1 + (left_y1 - avg_left_y1) / left_slope)
		left_y2 = bottom
		left_x2 = int(avg_left_x1 + (left_y2 - avg_left_y1) / left_slope)
		left_line = [left_x1, left_y1, left_x2, left_y2]

		
		last_left_line = left_line
		xL = get_cross_point(left_line[0], left_line[1], left_line[2], left_line[3]) 
		       
	except ValueError:
		repair = 1
		left_line = last_left_line
		xL = get_cross_point(left_line[0], left_line[1], left_line[2], left_line[3])
		       
	row = img.shape[0]
	col = img.shape[1]	
	mid = (xL+xR)//2
	critical = col // 5

	if((col//2 - critical < mid < col//2 + critical) and not repair):
		cv2.line(img, (right_line[0], right_line[1]) ,(right_line[2], right_line[3]), [0,255,0], thickness)
		cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), [0,255,0], thickness)
		check = 0
	
	else:
		cv2.line(img, (right_line[0], right_line[1]) ,(right_line[2], right_line[3]), [255,0,0], thickness)
		cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), [255,0,0], thickness)
		check = 1

	return check

def get_slope(x1,y1,x2,y2):
	return ((y2-y1)/(x2-x1))

def get_cross_point(x1,y1,x2,y2):
	m = get_slope(x1,y1,x2,y2)
	return (720-y1)/m + x1

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
	return cv2.addWeighted(initial_img, α, img, β, γ)

def draw_text(img):
	font = ImageFont.truetype('font/Wanted.ttf', 150)
	img_pil = Image.fromarray(img)
	draw = ImageDraw.Draw(img_pil)
	draw.text((270, 110), 'WARNING', font=font, fill=(255,0,0))
	img = np.array(img_pil)
	return img

	
if __name__ == "__main__":
	filename = input("請輸入欲辨識的影片檔名: ")
	clip = VideoFileClip('input/' + filename)
	output = 'output/(' + filename + ')_output.mp4'

	out_clip = clip.fl_image(process_image)
	out_clip.write_videofile(output, audio=False)

	pause = input()