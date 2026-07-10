import cv2
from camera import Camera
from object_detection import create_color_mask, find_largest_contour, get_contour_center

def main() -> None:
    camera = Camera()

    #rough starting range for a bright green object
    lower_green = (35, 80, 80)
    upper_green = (85, 255, 255)

    try:
        while True:
            frame = camera.get_frame()

            mask = create_color_mask(
                frame,
                lower_hsv=lower_green,
                upper_hsv=upper_green,
            )

            contour = find_largest_contour(mask)

            if contour is not None and cv2.contourArea(contour) > 500:
                center = get_contour_center(contour)

                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

                if center is not None:
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        f"{center}",
                        (center[0] + 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                area = cv2.contourArea(contour)
                x, y, width, height = cv2.boundingRect(contour)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (255, 0, 0),
                    2, 
                )

                cv2.putText(
                    frame,
                    f"Area: {area}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Camera", frame)
            cv2.imshow("Color Mask", mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()