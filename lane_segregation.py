import numpy as np
import cv2

def section(image):
	h,w = image.shape[0],image.shape[1]
	third = h//3
	lower = image[2*third:,:]
	upper = image[:2*third,:]
	return lower,upper

def edge_section(image):
	h,w = image.shape[0],image.shape[1]
	h_half = h//2
	w_fourth = w//4
	image[:h_half - 20,:w_fourth + 10] = 0
	image[:h_half ,3*w_fourth + 30:] = 0
	image[:,w-50:] = 0
	image[:,:20] = 0
	image[:3*h//4 - 100,w - 300:] = 0
	return image

def thresholding(image):
	
	blurred_image = cv2.GaussianBlur(image,(7,7),cv2.BORDER_DEFAULT)
	gray_image = cv2.cvtColor(blurred_image,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray_image,50,150,apertureSize = 3)
	cleared = edge_section(edges)
	kernel = np.ones((5,5), np.uint8)
	dilated = cv2.dilate(cleared,kernel,iterations = 2)
	cv2.imshow("asd",edges)
	cv2.waitKey(0)
	return dilated

def videowrite():
	image = cv2.imread("./docs/Lane_segregation/1.jpg")
	h,w,l = image.shape
	size = (w,h)
	video = cv2.VideoWriter("./Lane_segregation.mp4",cv2.VideoWriter_fourcc(*'mp4v'),30,size)
	print("generating video...")
	for i in range(0,206):
		img = cv2.imread("./docs/Lane_segregation/"+str(i)+".jpg")
		video.write(img)
	video.release()

def detection(image):
	lower_section,upper_section = section(image)
	image_threshold = thresholding(lower_section)

	h,w = image_threshold.shape[0],image_threshold.shape[1]
	half = w//2
	left_half = image_threshold[:,:half]
	right_half = image_threshold[:,half:]
	left_count = cv2.countNonZero(left_half)
	right_count = cv2.countNonZero(right_half)
	lower_left = lower_section[:,:half]
	lower_right = lower_section[:,half:]

	if left_count > right_count:
		contours_left = cv2.findContours(left_half,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
		cv2.fillPoly(lower_left,pts = contours_left[-2],color=(0,255,0))
		contours_right = cv2.findContours(right_half,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
		cv2.fillPoly(lower_right,pts = contours_right[-2],color=(0,0,255))

		lower = np.concatenate((lower_left,lower_right),axis = 1)
		output_image = np.concatenate((upper_section,lower),axis = 0)

		pts = np.array([[400,320],[560,320],[900,530],[100,530]])
		pts = pts.reshape((-1,1,2))
		isClosed = True
		cv2.polylines(output_image,[pts],isClosed,(255,0,0),2)
		overlay = output_image.copy()
		cv2.fillPoly(overlay,[pts],(0,0,255))
		output_image = cv2.addWeighted(output_image,0.9,overlay,0.1,0)

		cv2.imshow("processed",output_image)
		cv2.waitKey(100)
		return output_image
	else:
		contours_left = cv2.findContours(left_half,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
		cv2.fillPoly(lower_left,pts = contours_left[-2],color=(0,0,255))
		contours_right = cv2.findContours(right_half,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
		cv2.fillPoly(lower_right,pts = contours_right[-2],color=(0,255,0))

		lower = np.concatenate((lower_left,lower_right),axis = 1)
		output_image = np.concatenate((upper_section,lower),axis = 0)

		pts = np.array([[400,320],[560,320],[900,530],[100,530]])
		pts = pts.reshape((-1,1,2))
		isClosed = True
		cv2.polylines(output_image,[pts],isClosed,(255,0,0),2)
		overlay = output_image.copy()
		cv2.fillPoly(overlay,[pts],(0,0,255))
		output_image = cv2.addWeighted(output_image,0.9,overlay,0.1,0)

		cv2.imshow("processed",output_image)
		cv2.waitKey(100)
		return output_image

def main():
	cap = cv2.VideoCapture("./docs/whiteline/whiteline.mp4")
	if (cap.isOpened()== False):
		print("Error playing Stream")
	count = 0
	while (cap.isOpened()):
		ret,frame = cap.read()
		if ret == True:
			image = frame.copy()
			result = detection(image)
			cv2.imwrite("./docs/Lane_segregation/"+str(count)+".jpg",result)
			count +=1
			if 0xFF == ('q'):
				break
		else:
			break
	videowrite()
	cap.release()
	cv2.destroyAllWindows()  

if __name__ == '__main__':
	main()
