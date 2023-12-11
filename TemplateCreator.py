import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from scipy.interpolate import splprep, splev
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont



def load_image(file_path):
	try:
		# Open the JPEG image file
		image = Image.open(file_path)
		return image
	except IOError:
		print("Unable to load image")
		return None


def preprocess_image(image_path):
	# Read the image
	img = cv2.imread(image_path)
	cv2.imwrite(os.path.join(output_path, "original.jpg"), img)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	# edges = cv2.Canny(blurred, 50, 150)

	_, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)

	kernel = np.ones((3, 3), np.uint8)

	# Perform morphological operations (erosion followed by dilation)
	thresh = cv2.erode(thresh, kernel, iterations=15)
	thresh = cv2.dilate(thresh, kernel, iterations=15)

	contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Approximate the contours to find the corners
	epsilon = 0.04 * cv2.arcLength(contours[0], True)
	approx = cv2.approxPolyDP(contours[0], epsilon, True)


	# Apply perspective transform to get a top-down view
	paper_pts = np.float32([approx[1][0], approx[0][0], approx[3][0], approx[2][0]])
	width = 210  # A4 width in mm
	height = 297  # A4 height in mm
	dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
	
	matrix = cv2.getPerspectiveTransform(paper_pts, dst_pts)
	warped = cv2.warpPerspective(img, matrix, (width, height))

	# plt.subplot(121),plt.imshow(img,cmap = 'gray')
	# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	# plt.subplot(122),plt.imshow(warped,cmap = 'gray')
	# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	# plt.show()

	cv2.imwrite(os.path.join(output_path, jpg_file), warped)
	

def img2hole(image):
	enhancer = ImageEnhance.Contrast(image)
	enhanced_image = enhancer.enhance(3)

	width, height = image.size
	for x in range(width):
		for y in range(height):
			pixel = enhanced_image.getpixel((x, y))
			if pixel != (0, 0, 0):
				enhanced_image.putpixel((x, y), (255, 255, 255))  # Set as white
	dilated_image = enhanced_image.filter(ImageFilter.MaxFilter(size=5))
	blurred_image = dilated_image.filter(ImageFilter.GaussianBlur(radius=10))
	grayscale_image = blurred_image.convert('L')
	threshold_image = grayscale_image.point(lambda p: 0 if p < 255 else 255, '1')
	
	threshold_image.save(os.path.join(output_path, f"enhanced_holes_{jpg_file}"))

	img = cv2.imread(os.path.join(output_path, f"enhanced_holes_{jpg_file}"))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, im = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	contours, hierarchy  = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	points = []
	for contour in contours:
		# Get the center of each contour
		M = cv2.moments(contour)
		if M["m00"] != 0:
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			points.append([cX, cY])

	contour_image = cv2.drawContours(img, contours, -1, (0,255,0), 10)

	# plt.subplot(121),plt.imshow(im,cmap = 'gray')
	# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	# plt.subplot(122),plt.imshow(contour_image,cmap = 'gray')
	# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	# plt.show()

	return points

def img2Outline(image):
	enhancer = ImageEnhance.Contrast(image)
	enhanced_image = enhancer.enhance(3)
	# width, height = image.size
	# for x in range(width):
	# 	for y in range(height):
	# 		pixel = enhanced_image.getpixel((x, y))
	# 		# Check if the pixel is not black
	# 		if (0, 0, 0) <= pixel <= (5, 5, 5):
	# 			enhanced_image.putpixel((x, y), (255, 255, 255))  # Set as white
	# dilated_image = enhanced_image.filter(ImageFilter.MaxFilter(size=5))
	blurred_image = enhanced_image.filter(ImageFilter.GaussianBlur(radius=10))
	grayscale_image = blurred_image.convert('L')
	threshold_image = grayscale_image.point(lambda p: 0 if p < 255 else 255, '1')

	threshold_image.save(os.path.join(output_path, f"enhanced_outline_{jpg_file}"))

	img = cv2.imread(os.path.join(output_path, f"enhanced_outline_{jpg_file}"))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, im = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
	contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	epsilon = 0.005 * cv2.arcLength(contours[0], True)
	poly_contour = cv2.approxPolyDP(contours[0], epsilon, True)

	points = np.squeeze(poly_contour, axis=1)
	points = np.vstack([points, points[0]])
	x_list = points[:, 0]
	y_list = points[:, 1]
	xy_coords = [(x, y) for x, y in points]
	# tck, u = splprep(points.T, u=None, s=0, per=1)   # Spline fitting
	# u_new = np.linspace(u.min(), u.max(), 1000)
	# x_new, y_new = splev(u_new, tck, der=0)

	contour_image = cv2.drawContours(img, poly_contour, -1, (255,255,0), 10)
	plt.subplot(121),plt.imshow(blurred_image,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(enhanced_image,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	plt.show()

	return xy_coords, x_list, y_list

def compass(points):
	points_array = np.array(points)
	distances = np.linalg.norm(points_array[:, None] - points_array, axis=-1)
	max_distance_idx = np.unravel_index(np.argmax(distances), distances.shape)

	p1 = points[max_distance_idx[0]]
	p2 = points[max_distance_idx[1]]
	
	# Finding p3, farthest point from line formed by p1 and p2
	p3 = None
	max_dist = -1
	for idx, point in enumerate(points):
		if idx != max_distance_idx[0] and idx != max_distance_idx[1]:
			dist = np.abs(np.cross(np.array(p2) - np.array(p1), np.array(point) - np.array(p1))) / np.linalg.norm(
				np.array(p2) - np.array(p1))
			if dist > max_dist:
				max_dist = dist
				p3 = point

	# Create a line equation from p3 that is perpendicular to the line between p1 and p2
	slope_p1_p2 = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] - p1[0] != 0 else float('inf')
	if slope_p1_p2 == 0:
		slope_perpendicular = float('inf')
	elif slope_p1_p2 == float('inf'):
		slope_perpendicular = 0
	else:
		slope_perpendicular = -1 / slope_p1_p2

	# Using point-slope form to get the equation of the line passing through p3 and perpendicular to p1-p2 line
	# Equation: y - y1 = m(x - x1), where (x1, y1) is p3
	# y = mx - mx1 + y1
	p3_perpendicular_y_intercept = p3[1] - slope_perpendicular * p3[0]

	# Find p4, the next closest point to this perpendicular line after p3
	p4 = None
	min_distance_p3_to_p4 = float('inf')
	for idx, point in enumerate(points):
		if idx != max_distance_idx[0] and idx != max_distance_idx[1] and point != p3:
			# Calculate perpendicular distance from the point to the line using point-to-line distance formula
			distance = np.abs(slope_perpendicular * point[0] - point[1] + p3_perpendicular_y_intercept) / np.sqrt(
				slope_perpendicular ** 2 + 1)
			if distance < min_distance_p3_to_p4:
				min_distance_p3_to_p4 = distance
				p4 = point

	distance_p1_to_p3 = np.linalg.norm(np.array(p1) - np.array(p3))
	distance_p2_to_p3 = np.linalg.norm(np.array(p2) - np.array(p3))

	if distance_p1_to_p3 < distance_p2_to_p3:
		north = p1
		south = p2
		east = p3
		west = p4
	else:
		north = p2
		south = p1
		west = p3
		east = p4

	return (north,east,south,west)

if __name__ == "__main__":

	#TASKS:
	# Add ability to transform image of A4 paper to perfect dimensions with no distortion
	#ALSO LET PEOPLE INPUT THEIR OWN X, Y REFERENCE FOR SIZE OF PAPER, SO IT CAN BE ANY SIZE BASED OF THE PAPER AS A REFERENCE
	#Improve the smoothing of the drawings spline, try doing high resolution spline to smooth rather than straight to smooth from like 4 points
	# 
	path = "/home/benjamin/Documents/Projects/RAM/inputs/mum"
	files = os.listdir(path)
	jpg_files = [file for file in files if file.lower().endswith('.jpg')]
	scan = False

	for jpg_file in jpg_files:
		file_path = os.path.join(path, jpg_file)
		folder = os.path.splitext(jpg_file)[0]
		output_path = os.path.join(path, "output", folder)
		os.makedirs(output_path, exist_ok=True) 
		
		
		if scan == False:
			preprocess_image(file_path)
			drawing = load_image(os.path.join(output_path, jpg_file))
		else:
			drawing = load_image(file_path)


		outline, x_list, y_list = img2Outline(drawing)
		try: LHole,RHole = img2hole(drawing)
		except: print("Error: Please place two separate holes on the drawing.")

		a4_x, a4_y = drawing.size 
		pix2mmX, pix2mmY= 210/a4_x, 297/a4_y

		north,east,south,west = compass(outline)


		width = np.sqrt((east[0] - west[0])**2 + (east[1] - west[1])**2)
		height = np.sqrt((north[0] - south[0])**2 + (north[1] - south[1])**2)

		#Bottom holes
		BottomY = south[1] - (height * 0.20)
		PixelGap = 75

		tolerance = a4_y*0.15  # 15% tolerance on a4 y dimension

		# Find indices of points within the tolerance of userY
		indices_close = [i for i, y_val in enumerate(y_list) if abs(y_val - BottomY) < tolerance]

		BLHole = min(x_list[indices_close]) + PixelGap
		BRHole = max(x_list[indices_close]) - PixelGap

		#Middle holes
		MidY = south[1] - (height * 0.45)
		indices_close = [i for i, y_val in enumerate(y_list) if abs(y_val - BottomY) < tolerance]
		MLHole = min(x_list[indices_close]) + PixelGap
		MRHole = max(x_list[indices_close]) - PixelGap

		##################### IMAGE #####################
		output_image = Image.new("RGB", (a4_x, a4_y), color="white")
		draw = ImageDraw.Draw(output_image)

		# Draw the polygon on the copied image
		draw.line(outline, fill="blue", width=10)

		##################### HEIGHT & WIDTH INFO #####################
		draw.line(((north[0], north[1]),(south[0], south[1])), fill="green", width=10)
		draw.line(((east[0], east[1]),(west[0], west[1])), fill="red", width=10)

		font = ImageFont.truetype("DejaVuSans.ttf", 48)  # Change the font and size if needed

		draw.text((50, 150),f"Width = {np.around(width * pix2mmX)}mm",fill="black", font=font)
		draw.text((50, 300),f"Height = {np.around(height * pix2mmY)}mm",fill="black", font=font)

		##################### OUTLINE & STITCHLINE #####################
		# ax.plot(x_list, y_list, linestyle='solid', linewidth=15, color='black')
		# ax.plot(x_list, y_list, linestyle='dotted', linewidth=2, color='white')

		##################### BOTTOM HOLES #####################
		radius = 50
		
		draw.ellipse((BLHole - radius, BottomY - radius, BLHole + radius, BottomY + radius), fill="red")
		draw.ellipse((BRHole - radius,BottomY - radius, BRHole + radius,BottomY + radius), fill="red")
	
		##################### BOTTOM MIDDLE HOLES #####################
		draw.ellipse(((MLHole+BLHole)/2 - radius,(MidY+BottomY)/2 - radius, (MLHole+BLHole)/2 + radius,(MidY+BottomY)/2 + radius), fill="red")
		draw.ellipse(((MRHole+BRHole)/2 - radius,(MidY+BottomY)/2 - radius, (MRHole+BRHole)/2 + radius,(MidY+BottomY)/2 + radius), fill="red")

		# ##################### BOTTOM MIDDLE HOLES #####################
		draw.ellipse((MLHole - radius, MidY - radius, MLHole + radius, MidY + radius), fill="red")
		draw.ellipse((MRHole - radius, MidY - radius, MRHole + radius, MidY + radius), fill="red")

		# ##################### TOP HOLES #####################
		try:
			draw.ellipse((LHole[0] - radius, LHole[1] - radius, LHole[0] + radius, LHole[1] + radius), fill="red")
			draw.ellipse((RHole[0] - radius, RHole[1] - radius, RHole[0] + radius, RHole[1] + radius), fill="red")
		except: print("Couldnt find holes")
		output_image.save(f'{os.path.join(output_path, f"{jpg_file}")}')
		print(f'{os.path.join(output_path, f"{jpg_file}")}')
		# fig.savefig(f'{os.path.join(output_path, f"{jpg_file}")}', dpi=300, bbox_inches='tight')  # Set dpi as needed (300 is standard for printing)
		# plt.show()


