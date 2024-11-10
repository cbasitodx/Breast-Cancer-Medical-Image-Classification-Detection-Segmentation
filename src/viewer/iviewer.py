from abc import ABC, abstractmethod


class IViewer(ABC):
    @abstractmethod
    def plot_img(self):
        pass

    def plot_bounding_box(self):
        pass
