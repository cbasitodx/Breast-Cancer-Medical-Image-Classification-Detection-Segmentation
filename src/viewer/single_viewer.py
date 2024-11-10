from .iviewer import IViewer

import cv2 as cv
import numpy as np


class SingleViewer(IViewer):
    def __init__(self, file_name: str, dataset: str, title: str):
        # Get the image and label path
        self.__image_path = f"dataset/{dataset}/images/{file_name}.jpg"
        self.__label_path = f"dataset/{dataset}/labels/{file_name}.txt"

        self.__image = cv.imread(self.__image_path)
        self.__width, self.__height = self.__image.shape[:2]

        self.__title = title

        self.__tummor_coordinates: tuple | None = None

    def _plot_img(self):
        cv.imshow(self.__title, self.__image)
        self._wait_for_esc_or_close(self.__title)

    def _plot_bounding_box(self):
        cv.rectangle(self.__image, *self._get_tummor_coordinates(), (255, 0, 255), 2)

        # Make the rest of the img darker
        mask = np.zeros(self.__image.shape, np.uint8)
        cv.rectangle(mask, *self._get_tummor_coordinates(), (255, 255, 255), -1)

        darken_factor = 0.5
        inverted_mask = cv.bitwise_not(mask)
        darkened = cv.multiply(self.__image, np.array([darken_factor] * 3, dtype=float))

        final = cv.bitwise_and(darkened, inverted_mask) + cv.bitwise_and(self.__image, mask)

        cv.imshow(self.__title, final)
        self._wait_for_esc_or_close(self.__title)

    def _get_tummor_coordinates(self):
        if self.__tummor_coordinates is not None:
            return self.__tummor_coordinates

        with open(self.__label_path, 'r') as f:
            text = f.readline().strip()

        tummor_type, x, y, w, h = text.split()

        # Set the correct types
        tummor_type = int(tummor_type)
        x, y, w, h = map(float, [x, y, w, h])

        # Scale the float values to image dimensions
        x = int(x * self.__width)
        y = int(y * self.__height)
        w = int(w * self.__width)
        h = int(h * self.__height)

        # Get the top left values of x and y
        x = int(x - (w / 2))
        y = int(y - (h / 2))

        top_left = (x, y)
        bottom_right = (x + w, y + h)

        self.__tummor_coordinates = top_left, bottom_right

        return self.__tummor_coordinates

    def _wait_for_esc_or_close(self, title: str):
        while True:
            if cv.getWindowProperty(title, cv.WND_PROP_VISIBLE) < 1:
                break
            if cv.waitKey(1) == 27:
                break

        cv.destroyAllWindows()