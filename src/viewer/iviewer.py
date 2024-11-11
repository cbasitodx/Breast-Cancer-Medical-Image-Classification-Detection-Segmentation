from abc import ABC, abstractmethod


class IViewer(ABC):
    """
    ===========
    **Summary**
    ===========

        This class is an interface for the SingleViewer class.

    ==================
    **Public Methods**
    ==================

        * **plot_img(self):** This method plots the image.
        * **plot_bounding_box(self):** This method plots the image with the bounding box.
    """
    @abstractmethod
    def plot_img(self):
        pass

    def plot_bounding_box(self):
        pass
