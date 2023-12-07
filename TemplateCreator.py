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

def enhance_image(image, threshold=128):
	# Enhance the contrast of the image
	grayscale_image = image.convert('L')
	enhancer = ImageEnhance.Contrast(grayscale_image)
	enhanced_image = enhancer.enhance(5)  # Adjust the factor for desired contrast enhancement
	threshold_image = enhanced_image.point(lambda p: 0 if p < threshold else 255, '1')
	dilated_image = threshold_image.filter(ImageFilter.MaxFilter(size=3))

	plt.subplot(121),plt.imshow(image,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(dilated_image,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	plt.show()
	return enhanced_image

def img2point(image):
	pixel_data = np.array(image)

	# Get height and width of the image
	height, width = pixel_data.shape

	# Create lists to store x, y coordinates
	x_coords = []
	y_coords = []

	# Iterate through each pixel and create point cloud
	for y in range(height):
		for x in range(width):
			if pixel_data[y][x] == 0:  # Check if pixel is white (representing a line)
				x_coords.append(x)
				# Invert y-axis to match typical image coordinates
				y_coords.append(height - y)


	points = [(x, y) for x, y in zip(x_coords, y_coords)]

	# Apply the Ramer-Douglas-Peucker algorithm to simplify the polyline
	tolerance = 1.0  # Adjust this value as needed
	simplified_points = approximate_polygon(points, tolerance=tolerance)

	# Extract X & Y coordinates from the simplified points
	simplified_x_coordinates, simplified_y_coordinates = zip(*simplified_points)


	# Plot original points and fitted curve
	plt.scatter(x_coords, y_coords, s=1, color='black', label='Original Points')
	# plt.plot(x_curve, y_curve, color='red', label='Fitted Curve')

	plt.xlabel('X-axis')
	plt.ylabel('Y-axis')
	plt.title('Polynomial Regression - Circle Approximation')
	plt.gca().invert_yaxis()  # Invert y-axis to match image orientation
	plt.legend()
	plt.grid(True)
	plt.show()

def vectorize_image(image):
	img = cv2.imread(image)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	filtered = cv2.bilateralFilter(gray, 15, 75, 75)
	blur = cv2.GaussianBlur(filtered, (9, 9), 0)
	kernel = np.ones((10, 10), np.uint8) # Adjusting this helps
	morph = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
	edges = cv2.Canny(morph, 50, 150)
	kernel = np.ones((10, 10), np.uint8)
	dilated_edges = cv2.dilate(edges, kernel, iterations=1)
	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	smoothed_image = np.zeros_like(img)

	for contour in contours:
		epsilon = 0.001 * cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, epsilon, True)
		cv2.drawContours(smoothed_image, [approx], -1, (0, 255, 0), 2)

	plt.subplot(121),plt.imshow(img,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(dilated_edges,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	plt.show()

pic = load_image("/home/benjamin/Documents/Projects/RAM/right.jpg")

epic = enhance_image(pic)

# point = img2point(epic)


epic.save("cont.jpg")

vector = vectorize_image("mono.jpg")
# vector.save("vect.jpg")

