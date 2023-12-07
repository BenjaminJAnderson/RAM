from PIL import Image, ImageFilter
import cv2
import numpy as np

from matplotlib import pyplot as plt

def load_image(file_path):
	try:
		# Open the JPEG image file
		image = Image.open(file_path)
		return image
	except IOError:
		print("Unable to load image")
		return None

def enhance_image(image):
	grayscale_image = image.convert('L')
	threshold_image = grayscale_image.point(lambda p: 0 if p < 255 else 255, '1')
	dilated_image = threshold_image.filter(ImageFilter.MaxFilter(size=5))
	compatible_image = dilated_image.convert('RGB')
	blurred_image = compatible_image.filter(ImageFilter.GaussianBlur(radius=10))

	# plt.subplot(121),plt.imshow(threshold_image,cmap = 'gray')
	# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	# plt.subplot(122),plt.imshow(blurred_image,cmap = 'gray')
	# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	# plt.show()
	return blurred_image

def vectorize_image(image):
	img = cv2.imread(image)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, im = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
	contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	smooth_contours = []
	for contour in contours:
		epsilon = 0.01 * cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, epsilon, True)
		smooth_contours.append(approx)
	contour_image = np.zeros_like(img)
	contour_image = cv2.drawContours(img, smooth_contours, -1, (0,255,0), 10)



	plt.subplot(121),plt.imshow(im,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(contour_image,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	plt.show()

pic = load_image("/home/benjamin/Documents/Projects/RAM/left.jpg")

epic = enhance_image(pic)
epic.save("enhanced.jpg")
vector = vectorize_image("/home/benjamin/Documents/Projects/RAM/enhanced.jpg")


