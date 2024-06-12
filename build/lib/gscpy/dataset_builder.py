from .utils.file_manager import save_data_and_labels
from sys import setrecursionlimit
import matplotlib.pyplot as plt

setrecursionlimit(10000000)

class PointCollector:
    """An interactive 2D dataset builder, running on matplotlib."""
    def __init__(self,save):
        self.save=save
        self.points = {color: [] for color in self.color_map.values()}
        self.current_color = 'blue'
        self.is_drawing = False
        self.eraser_mode = False
        self.counter_mode = False
        self.fig, self.ax = plt.subplots()
        self.scatter_plots = {color: self.ax.plot([], [], 'o', color=color)[0] for color in self.color_map.values()}
        self.create_legend()

    @property
    def color_map(self):
        return {
            '1': 'blue', '2': 'green', '3': 'red', '4': 'cyan',
            '5': 'magenta', '6': 'yellow', '7': 'black',
            '8': 'orange', '9': 'purple'
        }

    def create_legend(self):
        self.legend_patches = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=6, label=f'{key}')
            for key, color in self.color_map.items()
        ]
        if self.save:
            self.save_patch = plt.Line2D([0], [0], linestyle="None", marker="", color='black', label="Press '0' to save")
        self.erase_patch = plt.Line2D([0], [0], linestyle="None", marker="", color='black', label="Press 'E' to enable eraser")
        self.erase_all_patch = plt.Line2D([0], [0], linestyle="None", marker="", color='black', label="Press 'R' to reset")
        self.counter_patch = plt.Line2D([0], [0], linestyle="None", marker="", color='black', label="Press 'C' to enable counter")

        # Place the color legend outside the plot at the top left of the figure, horizontally
        self.legend1 = self.fig.legend(handles=self.legend_patches, loc='upper left', bbox_to_anchor=(0.1, 0.95), markerscale=1, ncol=len(self.legend_patches))

        # Place the save and erase instruction legend outside the plot at the top right of the figure
        if self.save:
            feature_handles=[self.save_patch, self.erase_patch, self.erase_all_patch, self.counter_patch]
        else:
            feature_handles=[self.erase_patch, self.erase_all_patch, self.counter_patch]

        self.legend2 = self.fig.legend(handles=feature_handles, loc='upper right', bbox_to_anchor=(0.9, 1), frameon=False)
        self.erase_label = self.legend2.get_texts()[1]
        self.erase_all_label = self.legend2.get_texts()[2]
        self.counter_label = self.legend2.get_texts()[3]

        self.counter_patches = []
        self.legend3 = None

    def create_counter_legend(self):
        if self.legend3 is not None:
            self.legend3.remove()

        if self.counter_mode and any(len(points) > 0 for points in self.points.values()):
            self.counter_patches = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=6, label=f'{len(points)}')
                for color, points in self.points.items() if points
            ]
            self.legend3 = self.fig.legend(handles=self.counter_patches, loc='upper left', bbox_to_anchor=(0, 0.85), markerscale=1, ncol=1)
            self.fig.canvas.draw()

    def on_press(self, event):
        if event.inaxes:
            self.is_drawing = True
            self.modify_points(event)

    def on_release(self, event):
        self.is_drawing = False

    def on_motion(self, event):
        if self.is_drawing and event.inaxes:
            self.modify_points(event)

    def modify_points(self, event):
        if self.eraser_mode:
            self.erase_point(event)
        else:
            self.add_point(event)
        self.update_plot()

    def add_point(self, event):
        self.points[self.current_color].append((event.xdata, event.ydata))
        if self.counter_mode:
            self.create_counter_legend()

    def erase_point(self, event):
        erase_radius = 0.05
        for color, points in self.points.items():
            self.points[color] = [(x, y) for x, y in points if (x - event.xdata)**2 + (y - event.ydata)**2 > erase_radius**2]
        if self.counter_mode:
            self.create_counter_legend()

    def erase_all_points(self):
        self.points = {color: [] for color in self.color_map.values()}
        if self.counter_mode:
            self.create_counter_legend()
        self.update_plot()

    def update_plot(self):
        for color, line in self.scatter_plots.items():
            x_data, y_data = zip(*self.points[color]) if self.points[color] else ([], [])
            line.set_data(x_data, y_data)
        self.ax.draw_artist(self.ax.patch)
        for line in self.scatter_plots.values():
            self.ax.draw_artist(line)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()

    def generate_data_and_labels(self):
        data = []
        labels = []
        for color, points in self.points.items():
            data.extend(points)
            labels.extend([int(key) for key, val in self.color_map.items() if val == color] * len(points))
        return data, labels

    def change_color(self, key):
        if key in self.color_map:
            self.current_color = self.color_map[key]

    def toggle_eraser(self):
        self.eraser_mode = not self.eraser_mode
        if self.eraser_mode:
            self.erase_label.set_text("Press 'E' to disable zone eraser")
        else:
            self.erase_label.set_text("Press 'E' to enable zone eraser")
        self.fig.canvas.draw()

    def toggle_counter(self):
        self.counter_mode = not self.counter_mode
        if self.counter_mode:
            self.counter_label.set_text("Press 'C' to hide counter")
            self.create_counter_legend()
        else:
            self.counter_label.set_text("Press 'C' to show counter")
            if self.legend3 is not None:
                self.legend3.remove()
                self.legend3 = None
        self.fig.canvas.draw()

def build_dataset(save=False,name=None,path=None):
    """Runs an interactive 2D dataset builder, running on matplotlib.
     Inputs :
        save (bool, optional) : Set to True to save the dataset externally, in two files : name_data and name_labels.
         These files can then be loaded using gscpy.file_manager.load_data_and_labels . Default = False 
        name (str): If save is True, the name of your dataset, without file extension. Example: 'myset'.
        path (str): If save is True, the path to save the dataset. Example : 'Datasets/2D'.
        
     Returns :
        data (ndarray) :The points drawn.
        labels (ndarray) : The labels assigned to each point.
        """
    collector = PointCollector(save)

    collector.fig.canvas.mpl_connect('button_press_event', collector.on_press)
    collector.fig.canvas.mpl_connect('button_release_event', collector.on_release)
    collector.fig.canvas.mpl_connect('motion_notify_event', collector.on_motion)

    def on_key(event):
        if event.key in collector.color_map:
            collector.change_color(event.key)
        elif event.key == '0' and save:
            data, labels = collector.generate_data_and_labels()
            save_data_and_labels(data, labels, name,path)
        elif event.key.lower() == 'e':
            collector.toggle_eraser()
        elif event.key.lower() == 'r':
            collector.erase_all_points()
        elif event.key.lower() == 'c':
            collector.toggle_counter()

    collector.fig.canvas.mpl_connect('key_press_event', on_key)

    collector.ax.set_xticks([])  
    collector.ax.set_yticks([])  

    
    plt.show()
    
    data,labels=collector.generate_data_and_labels()
    return data,labels
    
