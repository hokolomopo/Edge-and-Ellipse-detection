import numpy as np
import matplotlib.pyplot as plt

from skimage import measure, io
from skimage.color import rgb2gray


# Construct some test data
# x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
# r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
image = io.imread("img/building.png")
r= rgb2gray(image)

# Find contours at a constant value of 0.8
contours = measure.find_contours(r, 0.5)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(r, cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
