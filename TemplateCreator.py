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

def enhance_image(image):
	grayscale_image = image.convert('L')
	threshold_image = grayscale_image.point(lambda p: 0 if p < 255 else 255, '1')
	dilated_image = threshold_image.filter(ImageFilter.MaxFilter(size=5))

	plt.subplot(121),plt.imshow(threshold_image,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(dilated_image,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	plt.show()
	return threshold_image

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
	ret, im = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
	contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	largest_contour = max(contours, key=cv2.contourArea)

	contour_image = np.zeros_like(img)
	contour_image = cv2.drawContours(img, contours, -1, (0,255,0), 10)

	# Extract contour coordinates
	contour_coords = largest_contour.reshape(-1, 2)  # Reshape to get x, y coordinates

	# Create a 2D array for the image vector
	min_x, min_y = np.min(contour_coords, axis=0)
	max_x, max_y = np.max(contour_coords, axis=0)
	shape_width = max_x - min_x + 1
	shape_height = max_y - min_y + 1

	vector_image = np.zeros((shape_height, shape_width), dtype=np.uint8)
	shifted_coords = contour_coords - [min_x, min_y]
	cv2.drawContours(vector_image, [shifted_coords], -1, 255, thickness=cv2.FILLED)




	plt.subplot(121),plt.imshow(im,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(contour_image,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	plt.show()

pic = load_image("/home/benjamin/Documents/Projects/RAM/left.jpg")

epic = enhance_image(pic)

# point = img2point(epic)


# vector = vectorize_image("/home/benjamin/Documents/Projects/RAM/Untitled1.jpg")
# vector.save("vect.jpg")

