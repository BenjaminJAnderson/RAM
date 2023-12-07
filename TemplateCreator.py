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

	epsilon = 0.01 * cv2.arcLength(contours[0], True)
	poly_contour = cv2.approxPolyDP(contours[0], epsilon, True)

	
	smooth_contours = []
	for points in np.squeeze(poly_contour):
			t = np.arange(len(points))
			poly_degree = 3  # Adjust the polynomial degree as needed
			coeffs = np.polyfit(t, points[:, 0], poly_degree), np.polyfit(t, points[:, 1], poly_degree)
			t_new = np.linspace(0, len(points) - 1, 100)
			x_smooth = np.polyval(coeffs[0], t_new)
			y_smooth = np.polyval(coeffs[1], t_new)
			smooth_contours.append(np.column_stack((x_smooth, y_smooth)).astype(np.int32))

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


