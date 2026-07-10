import cv2
import numpy as np

def create_color_mask(
    frame: np.ndarray,
    lower_hsv: tuple[int, int, int],
    upper_hsv: tuple[int, int, int],
) -> np.ndarray:

    """Return a binary mask containing pixels inside the selected HSV range."""

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_limit = np.array(lower_hsv, dtype=np.uint8)
    upper_limit = np.array(upper_hsv, dtype=np.uint8)

    mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)

    return mask

def find_largest_contour(mask: np.array):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    return max(contours, key=cv2.contourArea)

def get_contour_center(contour) -> tuple[int, int] | None:
    moments = cv2.moments(contour)

    if moments['m00'] == 0:
        return None

    center_x = int(moments['m10'] / moments['m00'])
    center_y = int(moments['m01'] / moments['m00'])

    return center_x, center_y