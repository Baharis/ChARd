from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap, to_rgb,\
    hsv_to_rgb, rgb_to_hsv
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
import numpy as np
import pandas as pd


def safe_normalize(array: np.ndarray) -> np.ndarray:
    """Normalize data to 0-1 range, return 0.5s if all data is equal"""
    mn, mx = np.min(array), np.max(array)
    return (array - mn) / (mx - mn) if mn < mx else np.full_like(array, 0.5)


@dataclass
class ChardSeries:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    e: np.ndarray = None

    def __len__(self):
        return len(self.a)

    @classmethod
    def from_raw(cls, csv_path: str) -> 'ChardSeries':
        with open(csv_path, 'r') as csv_file:
            df = pd.read_csv(csv_file, header=None)
        return cls(df[0].array, df[1].array, df[2].array)

    @classmethod
    def from_named(cls, csv_path: str, emphasis: str = None) -> 'ChardSeries':
        with open(csv_path, 'r') as csv_file:
            df = pd.read_csv(csv_file)
        e = e.array if (e := df.get(emphasis, None)) is not None else None
        return cls(df['a'].array, df['b'].array, df['c'].array, e=e)

    @classmethod
    def from_any(cls, csv_path: str, emphasis: str = None) -> 'ChardSeries':
        try:
            return cls.from_named(csv_path, emphasis=emphasis)
        except KeyError:
            return cls.from_raw(csv_path)

    @property
    def abc(self):
        return np.vstack([self.a, self.b, self.c]).T

    def normalized(self, to: Union[int, tuple] = None) -> 'ChardSeries':
        abc = self.a, self.b, self.c
        if isinstance(to, int):
            return ChardSeries(*[x / x[to] for x in abc], self.e)
        elif isinstance(to, list):
            return ChardSeries(*[x / y for x, y in zip(abc, to)], self.e)
        else:
            return self

    def colors(self, cm: Colormap) -> np.ndarray:
        e = safe_normalize(self.e) if self.e is not None \
            else np.full_like(self.a, 0.5)
        return cm(e)


class ChardAxes(PolarAxes):
    """A mix between Polar/Radial axes, with fixed 3 variables to plot"""
    name = 'chard'
    DEFAULT_COLORS = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    FONT_SIZE = 15
    THETA = np.linspace(0, 2 * np.pi, 3, endpoint=False)

    def __init__(self, *args, **kwargs) -> None:
        """Override starting location to be on the right"""
        super().__init__(*args, **kwargs)
        self.set_theta_zero_location('E')
        self.set_thetagrids(np.degrees(self.THETA), ['a', 'b', 'c'],
                            fontsize=self.FONT_SIZE)
        self.set_rlabel_position(90)
        self.set_axisbelow(False)
        self.grid(linewidth=1)
        self.r_min = 1.0
        self.r_max = 1.0

    @property
    def r_span(self):
        return self.r_max - self.r_min

    def plot(self, *args, **kwargs) -> Line2D:
        """Override plot so that line is closed by default"""
        lines = super().plot(self.THETA, *args, **kwargs)
        self._adapt_r_lims(lines)
        return self._close_lines(lines)

    def plot_series(self, cs: ChardSeries, color: str = None, **kwargs) -> Line2D:
        """Plot a series of y data, where y in 3xN- and emphasis is N-shaped"""
        colors = cs.colors(cm=self.generate_colormap(color))
        lines = []
        for abc, color in zip(cs.abc, colors):
            line = self.plot(abc, **kwargs)[0]
            line.set_color(color)
            lines.append(line)
        return lines

    def _adapt_r_lims(self, lines: Line2D) -> None:
        all_r = np.concatenate([line.get_ydata() for line in lines])
        self.r_min = min(self.r_min, min(all_r))
        self.r_max = max(self.r_max, max(all_r))
        self.set_rlim(self.r_min - 0.08 * self.r_span - 1e-8,
                      self.r_max + 0.02 * self.r_span + 1e-8)
        for label in self.get_yticklabels():
            label.set_bbox(dict(facecolor='white', edgecolor='None',
                                alpha=0.9, boxstyle='Round4, pad=0.1'))
            label.set_fontsize(self.FONT_SIZE)

    def _close_lines(self, lines: Line2D) -> Line2D:
        for line in lines:
            x, y = line.get_data()
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            line.set_data(x, y)
        return lines

    def _gen_axes_patch(self):
        # Axes patch centered at (0.5, 0.5), radius 0.5 in axes coordinates
        return Circle((0.5, 0.5), 0.5)

    def generate_colormap(self, color_or_colormap_name):
        try:
            cmap = plt.get_cmap(name=color_or_colormap_name)
        except ValueError:
            try:
                hsv = rgb_to_hsv(to_rgb(color_or_colormap_name))
            except KeyError:
                hsv = next(self.DEFAULT_COLORS)
            v_max_span = min([hsv[2], 1 - hsv[2]])
            hsv0 = [hsv[0], hsv[1], hsv[2] - 0.8 * v_max_span]
            hsv1 = [hsv[0], hsv[1], hsv[2] + 0.8 * v_max_span]
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
    ap.add_argument('-n', '--normalizer', action='append', default=[],
                    help='Index or values to normalize abc to, if needed; '
                         '"0" will normalize to 1st entry, '
                         '"12.4,6.5,30.4" will normalize to these values.')
    ap.add_argument('-e', '--emphasis', action='append', default=[],
                    help='Name of the column with information about emphasis; '
                         'Prefix the name with "!" to reverse the order')
    ap.add_argument('-o', '--output', action='store',
                    help='If given, save the figure under this name instead '
                         'of plotting it in an interactive mode.')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(i) for i in args.input]
    colors = cycle(ac if (ac := args.color) else [None])
    emphases = args.emphasis + [''] * len(input_paths)
    normalizers = []
    for an in args.normalizer:
        try:
            normalizers.append(int(an))
        except ValueError:
            normalizers.append([float(a.strip()) for a in an.split(',')])
    normalizers += [None] * len(input_paths)
    fig, ax = plt.subplots(subplot_kw=dict(projection='chard'))
    for input_path, color, emphasis, normalizer in \
            zip(input_paths, colors, emphases, normalizers):
        cs = ChardSeries.from_any(input_path, emphasis=emphasis)
        cs = cs.normalized(to=normalizer)
        ax.plot_series(cs, color=color)
    if args.output:
        plt.savefig(args.output, pad_inches=0.0)
    else:
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
