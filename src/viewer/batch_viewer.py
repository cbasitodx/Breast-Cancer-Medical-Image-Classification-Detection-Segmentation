from .single_viewer import SingleViewer


class BatchViewer:
    """
    ===========
    **Summary**
    ===========

        This class is used to visualize a batch of images and their bounding boxes.

    ==============
    **Parameters**
    ==============

        * **file_names:** (*list*) The names of the files without the extension.
        * **dataset:** (*str*) The dataset to get the images and labels from.
        * **titles:** (*str | list*) The titles of the windows. If it is a string, the title will be the string
            concatenated with the index of the image.

    ======================
    **Instance Variables**
    ======================

        * **__viewers:** (*list*) -- The list of SingleViewer objects.

    ===========
    **Example**
    ===========

        >>> viewer = BatchViewer(["1", "2"], "train", "Image")
        >>> viewer.plot_imgs()
        >>> viewer.plot_bounding_boxes()

    """

    def __init__(self, file_names: list, dataset: str, titles: str | list[str]):
        if isinstance(titles, str):
            titles = [f"{titles} {i}" for i in range(len(file_names))]

        self.__viewers = [SingleViewer(file_name, dataset, title) for file_name, title in zip(file_names, titles)]

    def plot_imgs(self):
        for viewer in self.__viewers:
            viewer.plot_img()

    def plot_bounding_boxes(self):
        for viewer in self.__viewers:
            viewer.plot_bounding_box()
