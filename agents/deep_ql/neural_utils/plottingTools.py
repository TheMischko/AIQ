import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    divider = np.expand_dims(l2, axis)
    return a / divider


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

    COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'brown', 'pink', 'gray', 'black']


    def __init__(self):
        self.average_arrs = list()

    def on_exit(self):
        if len(self.average_arrs) > 0:
            np_average_arrs = np.array(self.average_arrs)
            avg_of_arrays = np.average(np_average_arrs, 0)
            self.plot_array(avg_of_arrays, "Average loss figure")
            self.average_arrs.clear()

    def add_values_to_average_arr(self, values_arr):
        if len(values_arr) > 0:
            self.average_arrs.append(values_arr)

    def plot_array(self, y, x = None, title="Figure", type="-"):
        x_points = np.array([i for i in range(len(y))]) if x is None else np.array(x)
        y_points = np.array(y)

        plt.title(title)
        plt.plot(x_points, y_points, type)
        plt.show()

    def plot_multiple_array(self, arrays, title="Figure", type="-"):
        x_points = np.array([i for i in range(len(arrays[0]))])
        if len(x_points) < 20:
            return
        fig, axes = plt.subplots(nrows=len(arrays), ncols=1, figsize=(12, 8))

        for i, arr in enumerate(arrays):
            smoothen_vals = savgol_filter(np.array(arr), window_length=11, polyorder=2)
            axes[i].plot(x_points, smoothen_vals, color=self.COLORS[i], linestyle=type)

        #plt.tight_layout()
        plt.show()

