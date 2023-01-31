import matplotlib.pyplot as plt
import numpy as np


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance


@singleton
class PlottingTools(object):
    NUM_AVERAGE_OVER_INPUTS = 5

    def __init__(self):
        self.average_arrs = list()

    def add_values_to_average_arr(self, values_arr):
        self.average_arrs.append(values_arr)
        if len(self.average_arrs) >= self.NUM_AVERAGE_OVER_INPUTS:
            np_average_arrs = np.array(self.average_arrs)
            arrs_normalized = normalized(np_average_arrs, axis=0)
            avg_of_arrays = np.average(np_average_arrs, 0)
            self.plot_array(avg_of_arrays, "Average figure")
            self.average_arrs.clear()

    def plot_array(self, arr, title="Figure", type="-"):
        x_points = np.array([i for i in range(len(arr))])
        y_points = np.array(arr)

        plt.title(title)
        plt.plot(x_points, y_points, type)
        plt.show()
