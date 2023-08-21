import cv2
import numpy as np
from google.colab import drive
from google.colab.patches import cv2_imshow
import math

drive.mount('/content/drive')

def add_shadow(img, mask, angle=[0, 0, 0], location=[0.05, 0.05], alpha=0.5, blur=0.09):
    """
    :param img: manipulated image
    :param mask: full mask image (the size of mask has to be same with img)
    :param angle: (dx, dy, dz) shadow angle
    :param location: (y, x) shadow location (relative location)
    :param alpha: shadow brightness, 0. ~ 1.
    :param blur: shadow blur, 0.01 ~ 1.
    :return: manipulated image with shadow
    """

    def get_M(h, w, f, theta, phi, gamma, dx, dy, dz):
        # Projection 2D -> 3D matrix
        A1 = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1], [0, 0, 1]])
        # Projection 3D -> 2D matrix
        A2 = np.array([[f, 0, w / 2, 0], [0, f, h / 2, 0], [0, 0, 1, 0]])
        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([[1, 0, 0, 0], [0, np.cos(theta), -np.sin(theta), 0], [0, np.sin(theta), np.cos(theta), 0], [0, 0, 0, 1]])
        RY = np.array([[np.cos(phi), 0, -np.sin(phi), 0], [0, 1, 0, 0], [np.sin(phi), 0, np.cos(phi), 0], [0, 0, 0, 1]])
        RZ = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0], [np.sin(gamma), np.cos(gamma), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        R = np.dot(np.dot(RX, RY), RZ)  # Composed rotation matrix with (RX, RY, RZ)
        T = np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])  # Translation matrix

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))

    def rotate_along_axis(image, theta=0, phi=0, gamma=0, dy=0, dx=0, dz=0):
        (height, width) = image.shape[:2]

        # Get radius of rotation along 3 axes
        rtheta = theta * math.pi / 180.0
        rphi = phi * math.pi / 180.0
        rgamma = gamma * math.pi / 180.0

        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        d = np.sqrt(height ** 2 + width ** 2)
        focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz = focal

        # Get projection matrix
        mat = get_M(height, width, focal, rtheta, rphi, rgamma, dx, dy, dz)

        return cv2.warpPerspective(image.copy(), mat, (width, height))
    
    temp_mask = mask.copy().astype(np.uint8)
    temp_mask = cv2.cvtColor(temp_mask[:, :, :3], cv2.COLOR_BGR2GRAY) if len(temp_mask.shape) > 2 else temp_mask
    mx, my, mw, mh = cv2.boundingRect(temp_mask)

    diagonal = (mh ** 2 + mw ** 2) ** 0.5
    location_y = int(location[0] * diagonal)
    location_x = int(location[1] * diagonal)

    temp_mask = cv2.cvtColor(temp_mask, cv2.COLOR_GRAY2BGR) if len(temp_mask.shape) < 3 else temp_mask

    # Rotate mask
    mask_rotated = rotate_along_axis(temp_mask, angle[0], angle[1], angle[2], location_y, location_x)

    # Blur mask
    t_mask = np.where(mask_rotated, alpha, 1)
    kernel_shape = int(np.max([mh, mw]) * blur)
    if kernel_shape % 2 == 0:
        kernel_shape += 1
    t_mask = cv2.GaussianBlur(t_mask, (kernel_shape, kernel_shape), 0)

    # Add shadow
    output = (img * t_mask)
    output = np.where(temp_mask, img, output)
    output = output.astype('uint8')

    return output - img


test = cv2.imread('./src/test_img.png')
test_mask = cv2.imread('./src/test_mask.png')
black = cv2.imread('./src/Black.png')

black = cv2.resize(black, dsize = (530,640),interpolation = cv2.INTER_LINEAR)

result = add_shadow(test, test_mask)

cv2_imshow(result)
cv2_imshow(test)
cv2.waitKey(0)
cv2.destroyAllWindows()