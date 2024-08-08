from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from itertools import cycle, islice
from pathlib import Path
import os
from typing import Any, Callable, Iterable, List, TypeAlias, Union

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap, to_rgb,\
    hsv_to_rgb, rgb_to_hsv
from matplotlib.lines import Line2D
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
import numpy as np
import pandas as pd


# ~~~~~~~~~~~~~~~~~~~~~~~~~ CONSTANTS AND TYPE HINTS ~~~~~~~~~~~~~~~~~~~~~~~~ #


IntOrVector3: TypeAlias = Union[int, np.ndarray]
PathLike = Union[str, bytes, os.PathLike]


NONES = [None] * 1_000_000
EMPTIES = [''] * 1_000_000


# ~~~~~~~~~~~~~~~~~~~~~~~~~~ CONVENIENCE FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def conversion_raises(value: Any, fun: Callable, exception: Exception) -> bool:
    """Return True if `fun(value)` raises `exception`, False otherwise"""
    try:
        _ = fun(value)
    except exception:  # noqa
        return True
    else:
        return False


def pairwise(it: Iterable):
    """Iterate `it` pairwise, e.g. (1, 2, 3, 4) -> (1, 2), (2, 3), (3, 4)"""
    it = iter(it)
    window = deque(islice(it, 1), maxlen=2)
    for x in it:
        window.append(x)
        yield tuple(window)


def safe_normalize(array: np.ndarray) -> np.ndarray:
    """Normalize data to 0-1 range, return 0.5s if all data is equal"""
    mn, mx = np.min(array), np.max(array)
    return (array - mn) / (mx - mn) if mn < mx else np.full_like(array, 0.5)


def str2int_or_vector(s: str) -> IntOrVector3:
    """Convert str "2" into int 2 and str "1,2,3" into np.array([1, 2, 3])"""
    return np.array([float(si.strip()) for si in s.split(',')], dtype=float) \
        if conversion_raises(s, int, ValueError) else int(s)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CHARD AXES ETC ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@dataclass
class ChardSeries:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    e: np.ndarray = None

    def __len__(self):
        return len(self.a)

    @classmethod
    def from_raw_csv(cls,
                     path: PathLike,
                     **_) -> 'ChardSeries':
        with open(path, 'r') as csv_file:
            df = pd.read_csv(csv_file, header=None)
        return cls(df[0].array, df[1].array, df[2].array)

    @classmethod
    def from_named_csv(cls,
                       path: PathLike,
                       emphasis: str = None,
                       **_) -> 'ChardSeries':
        with open(path, 'r') as csv_file:
            df = pd.read_csv(csv_file)
        e = e.array if (e := df.get(emphasis, None)) is not None else None
        return cls(df['a'].array, df['b'].array, df['c'].array, e=e)

    @classmethod
    def from_raw_spreadsheet(cls,
                             path: PathLike,
                             sheet: str = None,
                             **_) -> 'ChardSeries':
        with open(path, 'r') as csv_file:
            sheet = 0 if sheet is None else sheet
            df = pd.read_excel(csv_file, sheet, header=None)
        return cls(df[0].array, df[1].array, df[2].array)

    @classmethod
    def from_named_spreadsheet(cls,
                               path: PathLike,
                               sheet: str = None,
                               emphasis: str = None) -> 'ChardSeries':
        with open(path, 'r') as csv_file:
            sheet = 0 if sheet is None else sheet
            df = pd.read_excel(csv_file, sheet, header=0)
            e = e.array if (e := df.get(emphasis, None)) is not None else None
        return cls(df['a'].array, df['b'].array, df['c'].array, e=e)

    @classmethod
    def from_any(cls, *args, **kwargs) -> 'ChardSeries':
        """Args of this method should match at least one of tested readers"""
        readers = [cls.from_raw_csv,
                   cls.from_named_csv,
                   cls.from_raw_spreadsheet,
                   cls.from_named_spreadsheet]
        for reader in reversed(readers):
            try:
                return reader(*args, **kwargs)
            except (KeyError, ValueError, TypeError):
                continue
        raise ValueError('None of the implemented readers could read the data')

    @property
    def abc(self):
        return np.vstack([self.a, self.b, self.c]).T

    def normalized(self, to: IntOrVector3 = None) -> 'ChardSeries':
        """Normalize self to `to`th element if int or `to`'s values if array"""
        abc = self.a, self.b, self.c
        if isinstance(to, int):
            return ChardSeries(*[x / x[to] for x in abc], self.e)
        elif isinstance(to, np.ndarray):
            return ChardSeries(*[x / y for x, y in zip(abc, to)], self.e)
        elif to is None:
            return self
        else:
            return ValueError(repr(to))

    def colors(self, cm: Colormap) -> np.ndarray:
        e = safe_normalize(self.e) if self.e is not None \
            else np.full_like(self.a, 0.5)
        return cm(e)


class ColormapGenerator:
    """This class generates colormaps based on descriptors for ChardAxes"""
    DEFAULT_COLORS = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    SEPARATOR = ':'

    @staticmethod
    def _is_int(s: str) -> bool:
        return not conversion_raises(s, int, ValueError)

    @staticmethod
    def _is_mpl_color(s: str) -> bool:
        return not conversion_raises(s, to_rgb, ValueError)

    @staticmethod
    def _is_mpl_colormap(s: str) -> bool:
        return not conversion_raises(s, plt.get_cmap, ValueError)

    class ColormapGeneratorException(Exception):
        pass

    def _join_colormaps(self,
                        colormaps: List[LinearSegmentedColormap],
                        start: float = 0.0,
                        stop: float = 1.0
                        ) -> LinearSegmentedColormap:
        colors = [colormaps[0](0.0)]
        for cm in colormaps:
            colors.extend(list(cm(np.linspace(0., 1., 256)[1:])))
        start_index = min(max(0, int(start * len(colors))), len(colors))
        stop_index = min(max(0, int(stop * len(colors))), len(colors))
        color_stack = np.vstack(colors[start_index:stop_index])
        return LinearSegmentedColormap.from_list('', color_stack)

    def _split_colormap_descriptor(self, s: str) -> List[str]:
        """Recursively split `descriptor` into valid color(map) or int parts"""
        if self._is_int(s) or self._is_mpl_color(s) or self._is_mpl_colormap(s):
            return [s]
        else:
            sep_positions = [i for i, c in enumerate(s) if c == self.SEPARATOR]
            for sep_pos in sep_positions:
                with suppress(self.ColormapGeneratorException):
                    return [*self._split_colormap_descriptor(s[:sep_pos]),
                            *self._split_colormap_descriptor(s[sep_pos+1:])]
        raise self.ColormapGeneratorException(f'Could not interpret "{s}"')

    def generate_colormap(self, descriptor: str) -> Colormap:
        """
        Generate a colormap given `descriptor`. The definition of
        the descriptor can be complex and utilize the following objects:
        - mpl 1 color name (C) - this will generate uniform 1-color colormap;
        - mpl colormap name (CM) - this will get named mpl colormap;
        Additionally, multiple mpl colors can be juxtaposed to make a colormap:
        - 'C:C' - this will generate 2-color hsv gradient from 1st to 2nd C;
        - 'C:C:C' - this will generate 3-color hsv gradient 1-2-3 C etc.;
        Each of the gradients described above can be then sliced using
        'gradient:start_percent:stop_percent' notation.
        Some examples of the accepted notation are listed below:
        - 'viridis:25:75' will yield the central half of the 'viridis' palette;
        - 'red:lime:blue:red' will generate a circular rainbow palette;
        - '#ff0000:#0000ff:0:50' will generate the first half of the red-blue
          spectrum; you can then use the second half for another ChARd series.
        """
        if not descriptor:
            default = next(self.DEFAULT_COLORS)
            desc_parts = [default]
        else:
            desc_parts = self._split_colormap_descriptor(descriptor)
        desc_range = []
        while self._is_int(desc_parts[-1]):
            desc_range.insert(0, float(desc_parts.pop()) / 100)
        if not desc_range:
            desc_range = [0.0, 1.0]
        colormaps = []
        if len(desc_parts) == 1:
            if self._is_mpl_color(desc_parts[0]):
                desc_parts.append(desc_parts[0])
            if self._is_mpl_colormap(desc_parts[0]):
                colormaps = [plt.get_cmap(desc_parts[0])]
        for dp1, dp2 in pairwise(desc_parts):
            if self._is_mpl_colormap(dp1):
                colormaps.append(plt.get_cmap(dp1))
            elif self._is_mpl_color(dp1) and self._is_mpl_color(dp2):
                hsv1 = rgb_to_hsv(to_rgb(dp1))
                hsv2 = rgb_to_hsv(to_rgb(dp2))
                hsv_space = np.linspace(hsv1, hsv2, 256, axis=1).T
                rgb_space = np.vstack([hsv_to_rgb(hsv) for hsv in hsv_space])
                cm = LinearSegmentedColormap.from_list('', rgb_space)
                colormaps.append(cm)
        return self._join_colormaps(colormaps, desc_range[0], desc_range[1])


class ChardAxes(PolarAxes):
    """A mix between Polar/Radial axes, with fixed 3 variables to plot"""
    name = 'chard'
    DEFAULT_COLORS = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    FONT_SIZE = 15
    THETA = np.linspace(0, 2 * np.pi, 3, endpoint=False)

    def __init__(self, *args, **kwargs) -> None:
        """Plot r=1, set starting loc. on the right, set abc theta & labels"""
        super().__init__(*args, **kwargs)
        super().plot(np.linspace(0, 2*np.pi, 720), np.ones(720), linewidth=2.5,
                     color=plt.rcParams['grid.color'], zorder=2.4)
        self.colormap_generator = ColormapGenerator()
        self.set_theta_zero_location('E')
        self.set_thetagrids(np.degrees(self.THETA), ['a', 'b', 'c'],
                            fontsize=self.FONT_SIZE)
        self.set_rlabel_position(90)
        self.set_axisbelow(False)
        self.grid(linewidth=1)
        self.r_min = 1.0
        self.r_max = 1.0

    @property
    def r_span(self) -> float:
        return self.r_max - self.r_min

    def plot(self, *args, **kwargs) -> List[Line2D]:
        """Override plot: set tight r limits, close the lines by default"""
        self.set_thetagrids(np.degrees(self.THETA), fontsize=self.FONT_SIZE)
        lines = super().plot(self.THETA, *args, **kwargs)
        self._adapt_r_lims(lines)
        return self._close_lines(lines)

    def plot_series(self,
                    cs: ChardSeries,
                    color: str = None,
                    **kwargs) -> List[Line2D]:
        """Plot a series of y data, where y in 3xN- and emphasis is N-shaped"""
        colors = cs.colors(cm=self.colormap_generator.generate_colormap(color))
        lines = []
        for abc, color in zip(cs.abc, colors):
            line = self.plot(abc, **kwargs)[0]
            line.set_color(color)
            lines.append(line)
        return lines

    def _adapt_r_lims(self, lines: Line2D) -> None:
        """Set r limits to stick close to plot data, incl. minimum @ center"""
        all_r = np.concatenate([line.get_ydata() for line in lines])
        self.r_min = min(self.r_min, min(all_r))
        self.r_max = max(self.r_max, max(all_r))
        self.set_rlim(self.r_min - 0.08 * self.r_span - 1e-8,
                      self.r_max + 0.02 * self.r_span + 1e-8)
        for label in self.get_yticklabels():
            label.set_bbox(dict(facecolor='white', edgecolor='None',
                                alpha=0.9, boxstyle='Round4, pad=0.1'))
            label.set_fontsize(self.FONT_SIZE)

    @staticmethod
    def _close_lines(lines: Line2D) -> Line2D:
        """Append the first point to the end of the line to close a triangle"""
        for line in lines:
            x, y = line.get_data()
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            line.set_data(x, y)
        return lines


register_projection(ChardAxes)


# ~~~~~~~~~~~~~~~~~~~~~~~~ PARSING COMMAND LINE INPUT ~~~~~~~~~~~~~~~~~~~~~~~ #


class ChardNamespace(Namespace):
    """Expected output of `parse_args` function"""
    input: List[PathLike]
    sheet: List[str]
    color: List[str]
    axes: str
    normalizer: List[str]
    emphasis: List[str]
    output: PathLike
    linewidth: str
    labelsize: str


def parse_args() -> ChardNamespace:
    """Parse provided arguments if program was run directly from the CLI"""
    ap = ArgumentParser(
        prog='chard',
        description='Plot ChARd plots based on external tabulated input',
        epilog='Author: Daniel Tchoń, baharis @ GitHub',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('-i', '--input', action='append', default=[],
                    help='Path to input file with a single series to plot')
    ap.add_argument('-s', '--sheet', action='append', default=[],
                    help='Name of the sheet with data if reading spreadsheet.')
    ap.add_argument('-c', '--color', action='append', default=[],
                    help='Color or colormap to be used for plotting series')
    ap.add_argument('-a', '--axes', action='store', default='simple',
                    choices=['simple', 'right'],
                    help='Style of the axes behind the data.')
    ap.add_argument('-n', '--normalizer', action='append', default=[],
                    help='Index or values to normalize abc to, if needed; '
                         '"0" will normalize to 1st entry, while '
                         '"12.4,6.5,30.4" will normalize to these values.')
    ap.add_argument('-e', '--emphasis', action='append', default=[],
                    help='Name of the column with information about emphasis, '
                         'plot as darkening of color or progress on colormap. '
                         'Prefix the name with "@" to reverse the order.')
    ap.add_argument('-o', '--output', action='store', type=str,
                    help='If given, save the figure under this name instead '
                         'of plotting it in an interactive mode.')
    ap.add_argument('-w', '--linewidth', action='store', type=float, default=1,
                    help='Width of the lines used to plot series.')
    ap.add_argument('-l', '--labelsize', action='store', type=float, default=15,
                    help='Size of the axis and grid line labels.')
    return ap.parse_args()


# ~~~~~~~~~~~~~~~~~~~~~~~~~ PLOTTING COMPLETE FIGURE ~~~~~~~~~~~~~~~~~~~~~~~~ #


def plot_chard(ns: ChardNamespace) -> None:
    """Generate the ChARd plot based on the kwargs provided in namespace"""

    inputs = [Path(i) for i in ns.input]
    colors = cycle(ac if (ac := ns.color) else [None])
    emphases = ns.emphasis + EMPTIES
    normalizers = [str2int_or_vector(n) for n in ns.normalizer] + NONES
    sheets = ns.sheet + NONES
    lw = float(ns.linewidth)

    fig, ax = plt.subplots(subplot_kw=dict(projection='chard'))
    ax.FONT_SIZE = int(ns.labelsize)
    for i, c, e, n, s in zip(inputs, colors, emphases, normalizers, sheets):
        cs = ChardSeries.from_any(path=i, sheet=s, emphasis=e.lstrip('@'))
        cs.e = -cs.e if e.startswith('@') else cs.e
        cs = cs.normalized(to=n)
        ax.plot_series(cs, color=c, lw=lw)
    if ns.output:
        plt.savefig(ns.output, bbox_inches='tight', pad_inches=0.1, dpi=300)
    else:
        plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~ COMMAND LINE ENTRY POINT ~~~~~~~~~~~~~~~~~~~~~~~~ #


def main() -> None:
    args = parse_args()
    plot_chard(args)


def example() -> None:
    raw_path = Path(__file__).parent / 'examples' / 'raw.csv'
    cs = ChardSeries.from_any(raw_path)
    cs.normalized(to=[10, 10, 10])
    fig, ax = plt.subplots(subplot_kw=dict(projection='chard'))
    ax.plot_series(cs, color='red')
    plt.show()


if __name__ == '__main__':
    main()
