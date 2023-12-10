import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
from scipy.interpolate import splprep, splev


def load_image(file_path):
	try:
		# Open the JPEG image file
		image = Image.open(file_path)
		return image
	except IOError:
		print("Unable to load image")
		return None

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
	width, height = image.size
	for x in range(width):
		for y in range(height):
			pixel = enhanced_image.getpixel((x, y))
			# Check if the pixel is not black
			if (0, 0, 0) <= pixel <= (5, 5, 5):
				enhanced_image.putpixel((x, y), (255, 255, 255))  # Set as white
	dilated_image = enhanced_image.filter(ImageFilter.MaxFilter(size=5))
	blurred_image = dilated_image.filter(ImageFilter.GaussianBlur(radius=10))
	grayscale_image = blurred_image.convert('L')
	threshold_image = grayscale_image.point(lambda p: 0 if p < 255 else 255, '1')

	threshold_image.save(os.path.join(output_path, f"enhanced_outline_{jpg_file}"))

	img = cv2.imread(os.path.join(output_path, f"enhanced_outline_{jpg_file}"))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, im = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
	contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# plt.subplot(121),plt.imshow(enhanced_image,cmap = 'gray')
	# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	# plt.subplot(122),plt.imshow(threshold_image,cmap = 'gray')
	# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	# plt.show()

	epsilon = 0.005 * cv2.arcLength(contours[0], True)
	poly_contour = cv2.approxPolyDP(contours[0], epsilon, True)

	points = np.squeeze(poly_contour)
	tck, u = splprep(points.T, u=None, s=0, per=1)   # Spline fitting
	u_new = np.linspace(u.min(), u.max(), 1000)
	x_new, y_new = splev(u_new, tck, der=0)

	return x_new, y_new


if __name__ == "__main__":
	path = "/home/benjamin/Documents/Projects/RAM/inputs/mine"
	files = os.listdir(path)
	jpg_files = [file for file in files if file.lower().endswith('.jpg')]

	for jpg_file in jpg_files:
		file_path = os.path.join(path, jpg_file)
		folder = os.path.splitext(jpg_file)[0]
		output_path = os.path.join(path, "output", folder)
		os.makedirs(output_path, exist_ok=True) 
		
		drawing = load_image(file_path)

		x_list,y_list = img2Outline(drawing)
		try: x,y = img2hole(drawing)
		except: print("Error: Please place two separate holes on the drawing.")

		N_index = np.argmin(y_list)
		E_index = np.argmax(x_list)
		S_index = np.argmax(y_list)
		W_index = np.argmin(x_list)
		NX, NY = x_list[N_index], y_list[N_index]
		EX, EY = x_list[E_index], y_list[E_index]
		SX, SY = x_list[S_index], y_list[S_index]
		WX, WY = x_list[W_index], y_list[W_index]

		#A4 conversion
		A4_x, A4_y = 2480, 3508
		pix2mmX, pix2mmY= 210/A4_x, 297/A4_y

		width = np.sqrt((EX - WX)**2 + (EY - WY)**2)
		height = np.sqrt((NX - SX)**2 + (NY - SY)**2)

		#Bottom holes
		BottomY = SY - (height * 0.20)
		tolerance = 4
		PixelGap = 75
		indices_close = [i for i, y_val in enumerate(y_list) if abs(y_val - BottomY) < tolerance]
		BLHole = x_list[indices_close[0]] + PixelGap
		BRHole = x_list[indices_close[-1]] - PixelGap

		#Middle holes
		MidY = SY - (height * 0.45)
		indices_close = [i for i, y_val in enumerate(y_list) if abs(y_val - MidY) < tolerance]
		MLHole = x_list[indices_close[0]] + PixelGap
		MRHole = x_list[indices_close[-1]] - PixelGap

		##################### SETTINGS #####################
		fig, ax = plt.subplots(figsize=(8.26772, 11.6929)) # A4 Paper
		fig.tight_layout()
		ax.axis('off')

		##################### IMAGE #####################
		ax.imshow(drawing)

		##################### HEIGHT & WIDTH INFO #####################
		ax.plot([NX, SX], [NY, SY], 'g--')
		ax.plot([EX, WX], [EY, WY], 'r--')
		ax.text(50, 150, f"Width = {np.around(width * pix2mmX)}mm", fontsize=7)
		ax.text(50, 300, f"Height = {np.around(height * pix2mmY)}mm", fontsize=7)


		##################### OUTLINE & STITCHLINE #####################
		ax.plot(x_list, y_list, linestyle='solid', linewidth=15, color='black')
		ax.plot(x_list, y_list, linestyle='dotted', linewidth=2, color='white')

		##################### BOTTOM HOLES #####################
		ax.plot([BLHole, BRHole],[BottomY, BottomY], "ro", markersize=10)

		##################### MIDDLE HOLES #####################
		ax.plot([MLHole, MRHole],[MidY, MidY], "ro", markersize=10)

		##################### TOP HOLES #####################
		ax.plot(x,y, "ro", markersize=10)

		fig.savefig(f'{os.path.join(output_path, f"{jpg_file}")}', dpi=300, bbox_inches='tight')  # Set dpi as needed (300 is standard for printing)
		# plt.show()


