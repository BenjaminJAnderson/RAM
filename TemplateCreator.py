from PIL import Image, ImageEnhance, ImageFilter, ImageOps
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

def enhance_contrast(image):
	# Enhance the contrast of the image
	enhancer = ImageEnhance.Contrast(image)
	enhanced_image = enhancer.enhance(5)  # Adjust the factor for desired contrast enhancement
	return enhanced_image

def convert_to_monotone(image, threshold=128):
	grayscale_image = image.convert("L")  # Convert to grayscale
	# Threshold the image to create a monotone image
	monotone_image = grayscale_image.point(lambda p: 0 if p < threshold else 255, '1')
	return monotone_image

def vectorize_image(image):
	img = cv2.imread(image)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edges = cv2.Canny(blurred, 50, 150)
	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	smoothed_image = np.zeros_like(img)

	for contour in contours:
		epsilon = 0.01 * cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, epsilon, True)
		cv2.drawContours(smoothed_image, [approx], -1, (0, 255, 0), 2)

	plt.subplot(121),plt.imshow(img,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(smoothed_image,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	plt.show()

	return image

# Function to convert the loaded image into a vector format
# def convert_to_vector(image):
	# Code to process the image and convert it into a vector format using edge detection or other techniques

# # Function to extract the outline of the foot from the vector image
# def extract_foot_outline(vector_image):
# 	# Code to analyze the vector image and extract the foot outline, possibly using image processing algorithms

# # Function to smooth the extracted foot outline
# def smooth_outline(foot_outline):
# 	# Code to apply a smoothing algorithm to the foot outline, removing jagged edges or imperfections

# # Function to conform the outline to a more aesthetic shape
# def conform_to_aesthetic_shape(smoothed_outline):
# 	# Code to reshape the outline to a more aesthetic form, using predefined templates or mathematical transformations

# # Function to generate the final sandal shape
# def generate_sandal_shape(aesthetic_outline, sole_shape):
# 	# Code to combine the aesthetic foot outline with a sole shape to create the final sandal shape

# # Function to output/save/display the resulting sandal shape
# def output_result(sandal_shape):
# 	# Code to save or display the resulting sandal shape for the user

pic = load_image("/home/benjamin/Documents/Projects/RAM/left.jpg")

epic = enhance_contrast(pic)

eepic = convert_to_monotone(epic)


epic.save("cont.jpg")
eepic.save("mono.jpg")


vector = vectorize_image("mono.jpg")
# vector.save("vect.jpg")

