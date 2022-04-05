import cv2
import numpy as np

def section(image):
	h,w = image.shape[0],image.shape[1]
	half = h//2
	upper = image[:half,:]
	lower = image[half:,:]	
	return lower,upper

def seperate_yellow_and_white(image):
	image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	low = (0,0,0)
	high = (179,45,96)
	mask = cv2.inRange(image_hsv,low,high)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel = np.ones((8,8),dtype = np.uint8))
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel = np.ones((20,20),dtype = np.uint8))
	contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		hull = cv2.convexHull(cnt)
		cv2.drawContours(mask,[hull],0,(255), -1)
	mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel = np.ones((5,5),dtype = np.uint8))
	road = cv2.bitwise_and(image, image,mask = mask)
	road_hsv = cv2.bitwise_and(image_hsv,image_hsv,mask = mask)
	low_val = (0,0,100)
	high_val = (179,255,255)
	mask2 = cv2.inRange(road_hsv, low_val,high_val)
	result = cv2.bitwise_and(image, image,mask = mask2)
	return result

def section2(image):
	h,w = image.shape[0],image.shape[1]
	h_third = h//3
	w_third = w//3
	image[:h_third + 30,2 * w_third:] = (0,0,0)
	image[:h_third + 100,(2 * w_third) + 50:] = (0,0,0)
	return image

def fit(coordinates):
	x = coordinates[:,0]
	x_new = list(set(x))
	y = coordinates[:,1]
	coefs = np.polynomial.polynomial.polyfit(x,y,2)
	fit = np.polynomial.polynomial.polyval(x_new,coefs)
	y_new = [int(fitt) for fitt in fit]
	coords = []
	for i in range(len(x_new)):
		coords.append((y_new[i],x_new[i]))
	return coords,coefs

def get_yellow_and_white_line(image):
	yellow = cv2.inRange(image,(20,80,80),(80,255,255))
	white = cv2.inRange(image,(220,220,220),(255,255,255))
	return yellow,white

def compute_homography(image):
	lower_section,upper_section = section(image)
	point_src = np.array([[580,95],[755,95],[1110,310],[250,310]])
	point_dst = np.array([[0,0],[400,0],[400,400],[0,400]])
	H,s = cv2.findHomography(point_src,point_dst)
	return H

def get_line_images(coordinates,color):
	img = np.zeros((720,400,3), np.uint8)
	for i in range(len(coordinates)-2):
		cv2.line(img,coordinates[i],coordinates[i+1],color,2)
	return img

def prediction(image,H):
	global prev_yellow_coordinates
	global prev_white_coordinates
	
	lower_section,upper_section = section(image)
	s_image = section2(lower_section.copy())
	mask = seperate_yellow_and_white(s_image)
	point_dst = (720,400)
	warped_road = cv2.warpPerspective(mask,H,point_dst)
	yellow,white = get_yellow_and_white_line(warped_road)
	white = cv2.erode(white,np.ones((5,5),np.uint8),iterations = 1)
	white = cv2.dilate(white,np.ones((5,5),np.uint8),iterations = 3)
	yellow = cv2.erode(yellow,np.ones((5,5),np.uint8),iterations = 1)
	yellow = cv2.dilate(yellow,np.ones((5,5),np.uint8),iterations = 2)
	coordinates_yellow = np.argwhere(yellow == 255)
	coordinates_white = np.argwhere(white == 255)

	if len(coordinates_yellow) < 2000:
		coordinates_yellow = prev_yellow_coordinates
	elif len(coordinates_yellow) > len(prev_yellow_coordinates):
		prev_yellow_coordinates = coordinates_yellow

	if len(coordinates_white) < 2000:
		coordinates_white = prev_white_coordinates
	elif len(coordinates_white) > len(prev_white_coordinates):
		prev_white_coordinates = coordinates_white

	coords_yellow,coefs_yellow = fit(coordinates_yellow)
	coords_white,coefs_white = fit(coordinates_white)
	
	x_yellow,y_yellow = coords_yellow.pop(len(coords_yellow)//2)[0],coords_yellow.pop(len(coords_yellow)//2)[1]
	x_white,y_white = coords_white.pop(len(coords_white)//2)[0],coords_white.pop(len(coords_white)//2)[1]

	radius_yellow = ((1+(2*coefs_yellow[2]*x_yellow)**2)**(3/2))/abs(2*coefs_yellow[2])
	radius_white = ((1+(2*coefs_white[2]*x_white)**2)**(3/2))/abs(2*coefs_white[2])

	white_line = get_line_images(coords_white,(0,255,0))
	yellow_line = get_line_images(coords_yellow,(255,0,0))

	yellow_plus_white = cv2.addWeighted(yellow_line,1,white_line,1,0)
	cv2.circle(yellow_plus_white,(x_yellow,y_yellow),3,(0,0,255),2)
	cv2.circle(yellow_plus_white,(x_white,y_white),3,(0,0,255),2)

	pts = np.array([[580,95],[755,95],[1110,310],[250,310]])
	pts = pts.reshape((-1,1,2))
	isClosed = True
	cv2.polylines(lower_section,[pts],isClosed,(0,255,0),2)
	overlay = lower_section.copy()
	cv2.fillPoly(overlay,[pts],(0,0,255))
	lower_section_final = cv2.addWeighted(lower_section,0.9,overlay,0.1,0)
	
	image1 = cv2.vconcat([upper_section,lower_section_final])
	final = cv2.hconcat([yellow_plus_white,image1])
	text1 = "Radius_Right = "+str(radius_white)+""
	text2 = "Radius_Left = "+str(radius_yellow)+""
	
	if (radius_white < radius_yellow):	
		cv2.putText(final,"Turn: Right",(6,710),cv2.FONT_HERSHEY_SIMPLEX,1,(20,100,100),1,cv2.LINE_AA,False)
		cv2.putText(final,text1,(6,670),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA,False)
		cv2.putText(final,text2,(6,630),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA,False)
	if (radius_white > radius_yellow):
		cv2.putText(final,"Turn: Left",(6,710),cv2.FONT_HERSHEY_SIMPLEX,1,(20,100,100),1,cv2.LINE_AA,False)
		cv2.putText(final,text1,(6,670),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA,False)
		cv2.putText(final,text2,(6,630),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA,False)

	cv2.imshow("output",warped_road)
	cv2.waitKey(10)
	return final

def videowrite():
	image = cv2.imread("./docs/turn_predict/1.jpg")
	h,w,l = image.shape
	size = (w,h)
	video = cv2.VideoWriter("./turn_predict.mp4",cv2.VideoWriter_fourcc(*'mp4v'),30,size)
	print("generating video...")
	for i in range(0,236):
		img = cv2.imread("./docs/turn_predict/"+str(i)+".jpg")
		video.write(img)
	video.release()

def main():
	cap = cv2.VideoCapture("./docs/turn_prediction/challenge.mp4")
	if (cap.isOpened() == False):
		print("Error playing Stream")
	count = 0
	while (cap.isOpened()):
		ret,frame = cap.read()
		if ret == True:
			image = frame.copy()
			if (count == 0):
				H = compute_homography(image)
			result = prediction(image,H)
			cv2.imwrite("./docs/turn_predict/"+str(count)+".jpg",result)
			count +=1
			if 0xFF == ('q'):
				break
		else:
			break
	cap.release()
	cv2.destroyAllWindows()  
	videowrite()

if __name__ == '__main__':
	prev_yellow_coordinates = []
	prev_white_coordinates = []
	main()
