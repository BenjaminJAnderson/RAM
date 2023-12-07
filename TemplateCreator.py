from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np
from scipy.interpolate import splprep, splev


from matplotlib import pyplot as plt

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
			# Check if the pixel is not black
			if pixel != (0, 0, 0):
				# If it's not black, set it to white
				enhanced_image.putpixel((x, y), (255, 255, 255))  # Set as white

	grayscale_image = enhanced_image.convert('L')
	threshold_image = grayscale_image.point(lambda p: 0 if p < 255 else 255, '1')
	dilated_image = threshold_image.filter(ImageFilter.MaxFilter(size=5))
	compatible_image = dilated_image.convert('RGB')
	blurred_image = compatible_image.filter(ImageFilter.GaussianBlur(radius=10))
	blurred_image.save("enhanced_holes.jpg")

	img = cv2.imread("enhanced_holes.jpg")
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, im = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
	contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	plt.subplot(121),plt.imshow(image,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(im,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	plt.show()


	return blurred_image

def img2Outline(image):
	grayscale_image = image.convert('L')
	threshold_image = grayscale_image.point(lambda p: 0 if p < 255 else 255, '1')
	dilated_image = threshold_image.filter(ImageFilter.MaxFilter(size=5))
	compatible_image = dilated_image.convert('RGB')
	blurred_image = compatible_image.filter(ImageFilter.GaussianBlur(radius=10))
	blurred_image.save("enhanced_outline.jpg")

	img = cv2.imread("enhanced_outline.jpg")
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, im = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
	contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	plt.subplot(121),plt.imshow(image,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(blurred_image,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	plt.show()

	epsilon = 0.008 * cv2.arcLength(contours[0], True)
	poly_contour = cv2.approxPolyDP(contours[0], epsilon, True)

	
	smooth_contours = []

	points = np.squeeze(poly_contour)
	tck, u = splprep(points.T, u=None, s=0, per=1)   # Spline fitting
	u_new = np.linspace(u.min(), u.max(), 1000)
	x_new, y_new = splev(u_new, tck, der=0)


	contour_image = np.zeros_like(img)
	contour_image = cv2.drawContours(img, smooth_contours, -1, (0,255,0), 10)

	return x_new, y_new


if __name__ == "__main__":
	drawing = load_image("left.jpg")

	x_list,y_list = img2Outline(drawing)
	# x,y = img2hole(drawing)

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

	width = np.around(np.sqrt((EX - WX)**2 + (EY - WY)**2))
	height = np.around(np.sqrt((NX - SX)**2 + (NY - SY)**2))

	#Bottom holes
	BottomY = SY - (height * 0.20)
	tolerance = 4
	PixelGap = 75
	indices_close = [i for i, y_val in enumerate(y_list) if abs(y_val - BottomY) < tolerance]
	BLHole = x_list[indices_close[0]] + PixelGap
	BRHole = x_list[indices_close[-1]] - PixelGap

	#Bottom holes
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
	ax.text(50, 150, f"Width = {width * pix2mmX}mm", fontsize=7)
	ax.text(50, 300, f"Height = {height * pix2mmY}mm", fontsize=7)


	##################### OUTLINE & STITCHLINE #####################
	ax.plot(x_list, y_list, linestyle='solid', linewidth=15, color='black')
	ax.plot(x_list, y_list, linestyle='dotted', linewidth=2, color='white')

	##################### BOTTOM HOLES #####################
	ax.plot([BLHole, BRHole],[BottomY, BottomY], "ro", markersize=10)

	##################### MIDDLE HOLES #####################
	ax.plot([MLHole, MRHole],[MidY, MidY], "ro", markersize=10)



	fig.savefig('modified_a4_figure.png', dpi=300, bbox_inches='tight')  # Set dpi as needed (300 is standard for printing)
	plt.show()


