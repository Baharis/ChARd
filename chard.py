import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
import numpy as np


class ChardAxes(PolarAxes):
    name = 'chard'
    THETA = np.linspace(0, 2 * np.pi, 3, endpoint=False)

    def __init__(self, *args, **kwargs):
        """Override starting location to be on the right"""
        super().__init__(*args, **kwargs)
        self.set_theta_zero_location('E')

    def plot(self, *args, **kwargs):
        """Override plot so that line is closed by default"""
        lines = super().plot(self.THETA, *args, **kwargs)
        for line in lines:
            x, y = line.get_data()
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            line.set_data(x, y)

    def set_varlabels(self, labels):
        self.set_thetagrids(np.degrees(self.THETA), labels)

    def _gen_axes_patch(self):
        # Axes patch centered at (0.5, 0.5), radius 0.5 in axes coordinates
        return Circle((0.5, 0.5), 0.5)


register_projection(ChardAxes)


def example_data():
    data = [
        ('Basecase', [
            [0.08, 0.01, 0.03],
            [0.07, 0.05, 0.04],
            [0.01, 0.02, 0.05],
            [0.02, 0.01, 0.07],
            [0.01, 0.01, 0.02]]),
    ]
    return data


if __name__ == '__main__':
    data = example_data()
    spoke_labels = data.pop(0)

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='chard'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['b', 'r', 'g', 'm', 'y']
    data = data[0]
    case_data = data[1]
    # ax.set_rgrids([0.2, 0.4, 0.6, 0.8])

    for d, color in zip(case_data, colors):
        ax.plot(d, color=color)
    ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    labels = ('Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5')
    legend = ax.legend(labels, loc=(0.9, .95),
                              labelspacing=0.1, fontsize='small')

    fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
             horizontalalignment='center', color='black', weight='bold',
             size='large')

    plt.show()