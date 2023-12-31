from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import List, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap, to_rgb,\
    hsv_to_rgb, rgb_to_hsv
from matplotlib.patches import Circle
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
import numpy as np
import pandas as pd


def safe_normalize(array: np.ndarray) -> np.ndarray:
    """Normalize data to 0-1 range, return 0.5s if all data is equal"""
    mn, mx = np.min(array), np.max(array)
    return (array - mn) / (mx - mx) if mn < mx else np.full_like(array, 0.5)


@dataclass
class ChardSeries:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    p: np.ndarray = None
    t: np.ndarray = None

    @classmethod
    def from_raw(cls, csv_path: str) -> 'ChardSeries':
        with open(csv_path, 'r') as csv_file:
            df = pd.read_csv(csv_file, header=None)
        return cls(df[0], df[1], df[2])

    @classmethod
    def from_named(cls, csv_path: str) -> 'ChardSeries':
        with open(csv_path, 'r') as csv_file:
            df = pd.read_csv(csv_file)
        return cls(df['a'], df['b'], df['c'],
                   p=df.get('p', None), t=df.get('t', None))

    @classmethod
    def from_any(cls, csv_path: str) -> 'ChardSeries':
        try:
            return cls.from_named(csv_path)
        except KeyError:
            return cls.from_raw(csv_path)

    @property
    def abc(self):
        return np.vstack([self.a, self.b, self.c]).T

    @property
    def e(self):
        return self.p if self.p is not None else self.t

    def colors(self, cm: Colormap = mpl.colormaps['binary']) -> List[str]:
        cm = mpl.colormaps['binary'] if cm is None else cm
        e = safe_normalize(self.e) if self.e else [0.5, ] * len(self.a)
        return cm(e)


class ChardAxes(PolarAxes):
    """A mix between Polar/Radial axes, with fixed 3 variables to plot"""
    name = 'chard'
    DEFAULT_COLORS = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    THETA = np.linspace(0, 2 * np.pi, 3, endpoint=False)

    def __init__(self, *args, **kwargs):
        """Override starting location to be on the right"""
        super().__init__(*args, **kwargs)
        self.set_theta_zero_location('E')
        self.set_thetagrids(np.degrees(self.THETA), ['a', 'b', 'c'])

    def plot(self, *args, **kwargs):
        """Override plot so that line is closed by default"""
        lines = super().plot(self.THETA, *args, **kwargs)
        return self._close_lines(lines)

    def plot_series(self, cs: ChardSeries, color: str = None, **kwargs):
        """Plot a series of y data, where y in 3xN- and emphasis is N-shaped"""
        colors = cs.colors(cm=self.get_colormap(color))
        lines = []
        for abc, color in zip(cs.abc, colors):
            line = self.plot(abc, **kwargs)[0]
            line.set_color(color)
            lines.append(line)
        return lines

    def _close_lines(self, lines):
        for line in lines:
            x, y = line.get_data()
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            line.set_data(x, y)
        return lines

    def _gen_axes_patch(self):
        # Axes patch centered at (0.5, 0.5), radius 0.5 in axes coordinates
        return Circle((0.5, 0.5), 0.5)

    def get_colormap(self, color_or_colormap_name):
        try:
            cmap = plt.get_cmap(name=color_or_colormap_name)
        except ValueError:
            try:
                hsv = rgb_to_hsv(to_rgb(color_or_colormap_name))
            except KeyError:
                hsv = next(self.DEFAULT_COLORS)
            v_max_span = min([hsv[2], 1 - hsv[2]])
            hsv0 = hsv
            hsv1 = hsv
            hsv0[2] = hsv[2] - 0.8 * v_max_span
            hsv1[2] = hsv[2] + 0.8 * v_max_span
            rgb_limits = [hsv_to_rgb(hsv0), hsv_to_rgb(hsv1)]
            cmap = LinearSegmentedColormap.from_list('', rgb_limits)
        return cmap


register_projection(ChardAxes)


def parse_args() -> Namespace:
    """Parse provided arguments if program was run directly from the CLI"""
    ap = ArgumentParser(
        prog='chard',
        description='Plot ChARd plots based on external tabulated input',
        epilog='Author: Daniel Tchoń, baharis @ GitHub'
    )
    ap.add_argument('-i', '--input', action='append', default=[],
                    help='Path to input file with a single series to plot')
    ap.add_argument('-c', '--color', action='append', default=[],
                    help='Color or colormap to be used for plotting series')
    ap.add_argument('-e', '--emphasis', action='append', default=[],
                    help='Name of the column with information about emphasis; '
                         'Prefix the name with "!" to reverse the order')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(i) for i in args.input]
    colors = cycle(ac if (ac := args.color) else [None])
    emphases = args.emphasis + [''] * len(input_paths)
    fig, ax = plt.subplots(subplot_kw=dict(projection='chard'))
    for input_path, color, emphasis in zip(input_paths, colors, emphases):
        cs = ChardSeries.from_any(input_path)
        ax.plot_series(cs, color=color)
    plt.show()


def example() -> None:
    raw_path = Path(__file__).parent / 'examples' / 'raw.csv'
    cs = ChardSeries.from_any(raw_path)
    fig, ax = plt.subplots(subplot_kw=dict(projection='chard'))
    ax.plot_series(cs)
    ax.set_title('Raw data')
    plt.show()


if __name__ == '__main__':
    main()
