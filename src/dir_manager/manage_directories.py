from os import mkdir as make_dir, walk
from shutil import copy2


def manage_directories() -> None:
    create_dirs()

    root_paths: list[str] = ["./dataset/test/", "./dataset/train/", "./dataset/valid/"]
    class_img_paths: list[str] = list(map(lambda x: x + "classification/images/", root_paths))  # eg. ./dataset/test/classification/images/

    files = list(map(get_files, lambda x: x + "images/", root_paths))

    cancer_bools: list[list[bool]] = list(map(get_cancer_bools, files, root_paths))

    # Copy img files to classification img directories
    for i in range(3):
        copy_files(
            files[i],
            root_paths[i] + "images/",
            class_img_paths[i],
            cancer_bools[i]
        )


def create_dirs() -> None:
    """
    Create classification directories and subdirectories.

    :return: A tuple containing the root paths and classification image paths.
    """
    # Create classification directories
    root_paths: list[str] = ["./dataset/test/", "./dataset/train/", "./dataset/valid/"]
    for path in root_paths:
        mkdir(path, "classification")

    # Create images and bounding boxes directories
    classification_paths: list[str] = list(map(lambda x: x + "classification/", root_paths))    # eg. ./dataset/test/classification/
    for path in classification_paths:
        mkdir(path, "bounding_boxes")
        mkdir(path, "images")

    # Create cancer and no_cancer directories
    class_img_paths: list[str] = list(map(lambda x: x + "images/", classification_paths))    # eg. ./dataset/test/classification/images/
    class_bb_paths: list[str] = list(map(lambda x: x + "bounding_boxes/", classification_paths))    # eg. ./dataset/test/classification/bounding_boxes/
    for img, bb in zip(class_img_paths, class_bb_paths):
        mkdir(img, "cancer")
        mkdir(img, "no_cancer")
        mkdir(bb, "cancer")
        mkdir(bb, "no_cancer")

    return


def get_files(path: str) -> list[str]:
    return get_files_names(path)


def get_cancer_bools(files: list[str], path: str) -> list[bool]:
    def has_cancer(filepath: str) -> bool:
        with open(filepath, "r") as file:
            return file.read(1) == "1"

    return [has_cancer(path + "labels/" + file + ".txt") for file in files]


def mkdir(path: str, directory: str) -> None:
    new_path: str = path + directory
    try:
        make_dir(new_path)
    except FileExistsError:
        print(f"Directory '{new_path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{new_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_files(path: str):
    files: list[str] = []

    for (dirpath, dirnames, filenames) in walk(path):
        files.extend(filenames)
        break

    return files


def remove_file_extension(file: str) -> str:
    return file.rsplit(".", 1)[0]


def get_files_names(path: str) -> list[str]:
    return [remove_file_extension(file) for file in get_files(path)]


def copy_files(files: list[str], origin: str, destination: str, cancer_bools: list[bool]) -> None:
    for i in range(len(files)):
        try:
            if cancer_bools[i]:
                copy2(f"{origin}{files[i]}.jpg", f"{destination}cancer/")
            else:
                copy2(f"{origin}{files[i]}.jpg", f"{destination}no_cancer/")
        except FileNotFoundError:
            print(f"File '{files[i]}.jpg' not found.")
        except FileExistsError:
            print(f"File '{files[i]}.jpg' already exists.")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    manage_directories()
