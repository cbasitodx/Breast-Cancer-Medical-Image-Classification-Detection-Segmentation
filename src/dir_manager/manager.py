from os import mkdir, walk
from shutil import copy2


class DirectoryManager:
    """
    ===========
    **Summary**
    ===========

        This class manages the directories of the dataset. It creates the directories needed for the project from the
        original directories of the dataset.

        The user must have the original dataset. It follows the structure:

        .. code-block:: shell

            dataset
            ├── test
            │   ├── images
            │   └── labels
            ├── train
            │   ├── images
            │   └── labels
            └── valid
                ├── images
                └── labels

        The new directories will be:

        .. code-block:: shell

            dataset
            ├── test
            │   ├── classification
            │   │   ├── bounding_boxes
            │   │   │   ├── cancer
            │   │   │   └── no_cancer
            │   │   └── images
            │   │       ├── cancer
            │   │       └── no_cancer
            │   ├── images
            │   └── labels
            ├── train
            │   ├── classification
            │   │   ├── bounding_boxes
            │   │   │   ├── cancer
            │   │   │   └── no_cancer
            │   │   └── images
            │   │       ├── cancer
            │   │       └── no_cancer
            │   ├── images
            │   └── labels
            └── valid
                ├── classification
                │   ├── bounding_boxes
                │   │   ├── cancer
                │   │   └── no_cancer
                │   └── images
                │       ├── cancer
                │       └── no_cancer
                ├── images
                └── labels

    ==============
    **Parameters**
    ==============

        * **dataset_folder_path:** (*str*) The path to the dataset folder.

    ======================
    **Instance Variables**
    ======================

        * **__root_paths:** (*list[str]*) -- The paths to the test, train, and valid directories.
        * **__n_datasets:** (*int*) -- The number of datasets.

    """

    def __init__(self, dataset_folder_path: str):
        self.__root_paths = [f"{dataset_folder_path}/test/", f"{dataset_folder_path}/train/", f"{dataset_folder_path}/valid/"]
        self.__n_datasets = len(self.__root_paths)

    def manage_directories(self) -> None:
        """
        ===========
        **Summary**
        ===========

            This method manages the directories of the dataset. It creates the classification directories and
            subdirectories and copies the images and labels to their respective directories.

        """

        self.__create_dirs()
        files = self.__get_files_names()
        cancer, no_cancer = self.__classify_imgs_by_label(files)
        self.__copy_files(cancer, no_cancer)

    def __create_dirs(self) -> None:
        """
        ===========
        **Summary**
        ===========

            This method creates the classification directories and subdirectories.
        """
        def make_dir(path: str, dir_name: str) -> None:
            new_path: str = path + dir_name
            try:
                mkdir(new_path)
            except FileExistsError:
                print(f"Directory '{new_path}' already exists.")
            except PermissionError:
                print(f"Permission denied: Unable to create '{new_path}'.")
            except Exception as e:
                print(f"An error occurred: {e}")

        # Create classification directories
        for path in self.__root_paths:
            make_dir(path, "classification")

        # Create images and bounding boxes directories
        classification_paths: list[str] = list(map(lambda x: x + "classification/", self.__root_paths))
        for path in classification_paths:
            make_dir(path, "bounding_boxes")
            make_dir(path, "images")

        # Create cancer and no_cancer directories
        classification_img_paths: list[str] = list(map(lambda x: x + "images/", classification_paths))
        classification_bb_paths: list[str] = list(map(lambda x: x + "bounding_boxes/", classification_paths))
        for img, bb in zip(classification_img_paths, classification_bb_paths):
            make_dir(img, "cancer")
            make_dir(img, "no_cancer")
            make_dir(bb, "cancer")
            make_dir(bb, "no_cancer")

    def __get_files_names(self) -> list[list[str]]:
        """
        ===========
        **Summary**
        ===========

            This method returns a list of the file names in the labels directory
            of `test`, `train`, and `valid`.
            This names will not have a file extension, as the imgs have the same
            name. That way, we can match the labels with the images.
        """
        def get_files(path: str) -> list[str]:
            files: list[str] = []
            for _, _, filenames in walk(path):
                for filename in filenames:
                    files.append(filename.rsplit(".", 1)[0])
            return files

        return [get_files(path + "labels/") for path in self.__root_paths]

    def __classify_imgs_by_label(self, files_names: list[list[str]]) -> tuple[list[list[str]], list[list[str]]]:
        """
        ===========
        **Summary**
        ===========

            This method classifies the images of the three datasets by their label.
            The labels are in the labels directory of each dataset.

        ==============
        **Parameters**
        ==============

            * **files_names:** (*list[list[str]]*) -- The files names in the labels directory of the three datasets.

        ===========
        **Returns**
        ===========

            (*tuple[list[list[str]], list[list[str]]*) -- The list of files classified by label. The first list is the
            list of files with cancer, and the second list is the list of files without cancer.
        """
        def has_cancer(filepath: str) -> bool:
            with open(filepath, "r") as file:
                return file.read(1) == "1"

        cancer = [[] for _ in range(self.__n_datasets)]
        no_cancer = [[] for _ in range(self.__n_datasets)]

        for i, dataset in enumerate(self.__root_paths):
            for file in files_names[i]:
                if has_cancer(dataset + "labels/" + file + ".txt"):
                    cancer[i].append(file)
                else:
                    no_cancer[i].append(file)

        return cancer, no_cancer

    def __copy_files(self, cancer: list[list[str]], no_cancer: list[list[str]]) -> None:
        """
        ===========
        **Summary**
        ===========

            This method copies the images and labels of the datasets to their respective classification directories.

        ==============
        **Parameters**
        ==============

            * **cancer:** (*list[list[str]]*) -- The list of files with cancer.
            * **no_cancer:** (*list[list[str]]*) -- The list of files without cancer.
        """
        for i, dataset in enumerate(self.__root_paths):
            img_path = dataset + "classification/images/"

            for file in cancer[i]:
                copy2(dataset + "images/" + file + ".jpg", img_path + "cancer/")

            for file in no_cancer[i]:
                copy2(dataset + "images/" + file + ".jpg", img_path + "no_cancer/")
