from PIL import Image, ImageFilter
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

def img2Array(image):
	img = cv2.imread(image)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, im = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
	contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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


outline = load_image("/home/benjamin/Documents/Projects/RAM/right2.jpg")

enhanced_outline = enhance_image(outline)
enhanced_outline.save("enhanced.jpg")
x,y = img2Array("/home/benjamin/Documents/Projects/RAM/enhanced.jpg")

N_index = np.argmin(y)
NX, NY = x[N_index], y[N_index]

E_index = np.argmax(x)
EX, EY = x[E_index], y[E_index]

S_index = np.argmax(y)
SX, SY= x[S_index], y[S_index]

W_index = np.argmin(x)
WX, WY = x[W_index], y[W_index]

#A4 conversion
pix2mmX = 210/2480
pix2mmY = 297/3508

width = np.around(np.sqrt((EX - WX)**2 + (EY - WY)**2) * pix2mmX)
height = np.around(np.sqrt((NX - SX)**2 + (NY - SY)**2) * pix2mmY)


fig, ax = plt.subplots(figsize=(8.26772, 11.6929)) # A4 Paper
fig.tight_layout()
ax.axis('off')

ax.imshow(outline)
ax.plot([NX, SX], [NY, SY], 'g--')
ax.plot([EX, WX], [EY, WY], 'r--')

# plt.imshow(outline)
# plt.plot([NX,SX], [NY,SY], 'g--')
# plt.plot([EX,WX], [EY,WY], 'r--')

ax.plot(x, y, linestyle='solid', linewidth=15, color='black')
ax.plot(x, y, linestyle='dotted', linewidth=2, color='white')

# plt.plot(x, y, linestyle='solid', linewidth=10, color='black')
# plt.plot(x, y, linestyle='dotted',linewidth=2, color='white')

ax.text(50, 150, f"Width = {width}mm", fontsize=7)
ax.text(50, 300, f"Height = {height}mm", fontsize=7)

# plt.text(50,150,f"Width = {width}mm", fontsize=7)
# plt.text(50,300,f"Height = {height}mm", fontsize=7)

ax.set_xticks([]), ax.set_yticks([])

fig.savefig('modified_a4_figure.png', dpi=300, bbox_inches='tight')  # Set dpi as needed (300 is standard for printing)
plt.show()

# plt.xticks([]),plt.yticks([])

# plt.show()


