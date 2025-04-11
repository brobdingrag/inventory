# Standard library
import gzip
import hashlib
import json
import io
import webbrowser
import os
import sys
import re
import time
import functools
import unicodedata

# Specific standard library imports
from itertools import combinations
from collections import defaultdict, Counter
from textwrap import dedent

# Third party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyperclip
import requests

# Specific third party imports
from scipy.stats import norm, iqr
from tqdm import tqdm
import matplotlib.ticker as mtick
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator
from scipy.constants import golden
from IPython.display import clear_output

# LOCAL_PATHS
GENERAL_PATH = os.path.expanduser("~/general/")
EMAIL_PATH = os.path.expanduser("~/Email/")

def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)

ensure_dir(GENERAL_PATH)
ensure_dir(EMAIL_PATH)

# Colors
BLUE, LIGHT_BLUE, LIGHT_RED, RED = "#0000b3", "#5F5FFF", "#FF5F5F", "#b30000"
ORANGE = '#FFA500'
GREY, WHITE = "#808080", "#FFFFFF"
GREEN, LIGHT_GREEN = "#008000", "#5FBF5F"

MALE, FEMALE = "#1f77b4", "#e377c2"

# Pandas settings
pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 30)
pd.options.mode.chained_assignment = None
pd.set_option('display.max_colwidth', 80)

# Matplotlib settings
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Bitstream Vera Sans']
# plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['errorbar.capsize'] = 3.0

def download_file(url, file_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

# Jupyter notebook
def print_tmp(*s):
    """Jupyter notebook print function that clears the output first."""
    clear_output(wait=True)
    print(*s)

def autoreload():
    autoreload = """
    %load_ext autoreload
    %autoreload 2
    """
    copy(dedent(autoreload).lstrip("\n"))


## Number formatting
def percent(number, decimals=3):
    return f"{number:.{decimals}%}"

def comma(number, decimals=0):
    return f"{number:,.{decimals}f}"

def sign(x, decimals=2):
	if x == 0:
		return "0"
	elif x > 0:
		return f"$+${x:.{decimals}f}"
	else:
		return f"$-${abs(x):.{decimals}f}"

def copy_to_general(filepath, force=False):
    destination = os.path.join(GENERAL_PATH, os.path.basename(filepath))
    if not os.path.exists(destination) or force:
        os.system(f"cp {filepath} {destination}")

def email(filepath):
    """Shorter alias for email_file."""
    email_file(filepath)

def email_file(filepath):
    os.system(f"cp '{filepath}' {EMAIL_PATH}")

def open_file(filepath):
    os.system(f"open '{filepath}'")

def code_file(filepath):
    os.system(f"code {filepath}")

def copy_file(filepath, destination):
    os.system(f"cp {filepath} {destination}")

def move_file(filepath, destination):
    os.system(f"mv {filepath} {destination}")

def remove_file(filepath):
    os.system(f"rm {filepath}")

def make_dir(filepath):
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)

def copy_path():
    copy(f"{os.getcwd()}/")

def open_pdf(filepath):
    if not filepath.endswith(".pdf"):
        filepath = f"{filepath}.pdf"
    os.system(f"open {filepath}")

## Matplotlib

class Jax(Axes):

    nticks = 5
    labelpad = 7

    def __init__(self, ax):
        # Adopt the existing ax's attributes
        self.__dict__ = ax.__dict__

    def remove(self):
        self.set_visible(False)

    def legend_side(self, title=None, bbox_to_anchor=(1.05, 1), fontsize=8, loc='upper left', **kwargs):
        self.legend(bbox_to_anchor=bbox_to_anchor, loc=loc, title=title, fontsize=fontsize, **kwargs)

    def remove_legend(self):
        self.legend_.remove()

    def reverse_ylim(self):
         ymin, ymax = self.get_ylim()
         self.set_ylim(ymax, ymin)
     
    def reverse_xlim(self):
         xmin, xmax = self.get_xlim()
         self.set_xlim(xmax, xmin)
 
    # Remove ticks
    def remove_xticks(self):
        self.set_xticks([])
        
    def remove_yticks(self):
        self.set_yticks([])    

    # Number of ticks
    def set_nxticks(self, n=None):
        if n is None:
            n = self.nticks
        self.xaxis.set_major_locator(plt.MaxNLocator(n))

    def set_nyticks(self, n=None):
        if n is None:
            n = self.nticks
        self.yaxis.set_major_locator(plt.MaxNLocator(n))

    def set_nticks(self, n=None):
        self.set_nxticks(n)
        self.set_nyticks(n)

    # Minor ticks
    def set_minor_nxticks(self, n):
        self.xaxis.set_minor_locator(plt.MaxNLocator(n))

    def set_minor_nyticks(self, n):
        self.yaxis.set_minor_locator(AutoMinorLocator(n))

    # Labels
    def set_xlabel(self, xlabel, labelpad=None, **kwargs):
        if labelpad is None:
            labelpad = self.labelpad
        Axes.set_xlabel(self, xlabel, labelpad=labelpad, **kwargs)

    def set_ylabel(self, ylabel, labelpad=None, **kwargs):
        if labelpad is None:
            labelpad = self.labelpad
        Axes.set_ylabel(self, ylabel, labelpad=labelpad, **kwargs)

    # Remove labels
    def remove_xlabel(self):
        self.set_xlabel(None)

    def remove_ylabel(self):
        self.set_ylabel(None)

    # Floating spines
    def float_x(self, x):
        self.spines.bottom.set_bounds(min(x), max(x))

    def float_y(self, y):
        self.spines.left.set_bounds(min(y), max(y))
    
    # Naked 
    def naked(self):
        self.naked_top()
        self.naked_bottom()

    def naked_bottom(self):
        self.remove_bottom_spine()
        self.remove_xlabel()
        self.remove_xticks()

    def naked_top(self):
        self.remove_top_right_spines()
        self.remove_left_spine()
        self.remove_yticks()
        self.remove_ylabel()
    
    # Remove spines
    def remove_top_right_spines(self):
        self.remove_top_spine()
        self.remove_right_spine()

    def remove_top_spine(self):
        self.spines['top'].set_visible(False)

    def remove_right_spine(self):
        self.spines['right'].set_visible(False)

    def remove_left_spine(self):
        self.spines['left'].set_visible(False)

    def remove_bottom_spine(self):
        self.spines['bottom'].set_visible(False)

    def remove_spine(self, spine):
        self.spines[spine].set_visible(False)

    # Grid
    def grid(self, linestyle='-', alpha=0.4, which='major', **kwargs):
        Axes.grid(self, linestyle=linestyle, alpha=alpha, which=which, **kwargs)

    def xgrid(self, linestyle='--', alpha=0.4, which='major', **kwargs):
        self.grid(axis='x', linestyle=linestyle, alpha=alpha, which=which, **kwargs)    

    def ygrid(self, linestyle='-', alpha=0.15, which='major', **kwargs):
        self.grid(axis='y', linestyle=linestyle, alpha=alpha, which=which, **kwargs)    

    # Scales
    def log_xscale(self):
        self.set_xscale('log')

    def log_yscale(self):
        self.set_yscale('log')

    def hline(self, y, **kwargs):
        self.axhline(y, **kwargs)

    def vline(self, x, **kwargs):
        self.axvline(x, **kwargs)

    # Plus and minus signs
    def sign_xscale(self, decimals=0):
        self.xaxis.set_major_formatter(mtick.FuncFormatter(self.create_sign_formatter(decimals)))

    def sign_yscale(self, decimals=0):
        self.yaxis.set_major_formatter(mtick.FuncFormatter(self.create_sign_formatter(decimals)))

    # Commas
    def comma_xscale(self, decimals=0):
        self.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))

    def comma_yscale(self, decimals=0):
        self.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Percent 
    def percent_xscale(self, decimals=0):
        self.set_xlim(0, 1)
        self.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=decimals))

    def percent_yscale(self, decimals=0):
        self.set_ylim(0, 1)
        self.set_yticks([0, 0.25, 0.5, 0.75, 1])
        self.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=decimals))

    def twinx(self, **kwargs):
        return Jax(Axes.twinx(self, **kwargs))

    # Mirror
    def mirror_y(self, fontsize=None):
        ax_right = self.twinx()
        ax_right.set_yticks(self.get_yticks())
        ax_right.set_yticklabels(self.get_yticklabels(), fontsize=fontsize)
        ax_right.set_ylim(self.get_ylim())

    def mirror_x(self, fontsize=None):
        ax_top = self.twiny()
        ax_top.set_xticks(self.get_xticks())
        ax_top.set_xticklabels(self.get_xticklabels(), fontsize=fontsize)
        ax_top.set_xlim(self.get_xlim())

    def make_golden(self, orient='h'):
        assert orient in ['h', 'v']
        if orient == 'h':
            self.set_box_aspect(1 / golden)
        elif orient == 'v':
            self.set_box_aspect(golden)

    def make_square(self):
        self.set_box_aspect(1)

def get_fig(plt):
    return plt.gcf()

# General matplotlib
def plot_scatter(x, y, xlabel=None, ylabel=None, grid=True, linewidth=0.7, facecolors="none", alpha=0.8, 
                edgecolors="k", s=None, **kwargs):
    if not xlabel:
        if isinstance(x, pd.Series):
            xlabel = x.name
        else:
            xlabel = "x"
    if not ylabel:
        if isinstance(y, pd.Series):
            ylabel = y.name
        else:
            ylabel = "y"
    fig, ax = golden_fig()

    if s is None:
        s = 25

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.remove_top_right_spines()
    ax.scatter(x, y, s=s, facecolors=facecolors, edgecolors=edgecolors, linewidth=linewidth, alpha=alpha, **kwargs)
    ax.set_nyticks(5)
    ax.set_nxticks(5)
    if grid:
        ax.grid()
    return fig, ax

def square_fig(nrows=1, ncols=1, scale=1):
    height = 4.45  # close to np.sqrt(3.5**2 * golden)
    width = ncols *  height 
    height = nrows * height 
    fig, axes = plt.subplots(nrows, ncols, figsize=tuple(scale * np.array([width, height])))
    if type(axes) == np.ndarray:
        axes = axes.flatten()
        axes = [Jax(ax) for ax in axes]
        for ax in axes:
            # We can undo this later if we want different dimensions for the subplots
            ax.make_square()
            # ax.set_nticks()
    else:
        axes = Jax(axes)
        axes.make_square()
        axes.set_nticks()
    return fig, axes

def golden_fig(nrows=1, ncols=1, scale=1, orient='h'):
    assert orient in ['h', 'v']
    height = 3.5
    width = ncols * height 
    height = nrows * height 
    if orient == 'h':
        width *= golden
    elif orient == 'v':
        height *= golden
    fig, axes = plt.subplots(nrows, ncols, figsize=tuple(scale * np.array([width, height])))
    if type(axes) == np.ndarray:
        axes = axes.flatten()
        axes = [Jax(ax) for ax in axes]
        for ax in axes:
            ax.make_golden()
    else:
        axes = Jax(axes)
        axes.make_golden()
    return fig, axes

def clean_zip_filepath(filepath):
    if filepath.startswith("./"):
        filepath = filepath[2:]
    if not filepath.endswith(".zip"):
        filepath = filepath + ".zip"
    return filepath

def clean_pdf_filepath(filepath):
    if filepath.startswith("./"):
        filepath = filepath[2:]
    if not filepath.endswith(".pdf"):
        filepath = filepath + ".pdf"        
    return filepath

def clean_image_filepath(filepath):
    if filepath.startswith("./"):
        filepath = filepath[2:]
    if filepath.startswith("images/"):
        filepath = filepath[7:]
    return filepath

def clean_data_filepath(filepath):
    if filepath.startswith("./"):
        filepath = filepath[2:]
    if filepath.startswith("data/"):
        filepath = filepath[5:]
    return filepath

def code_fig(filepath, prefix='images/'):
    filepath = clean_image_filepath(filepath)
    if not filepath.endswith(".png"):
        filepath = filepath + ".png"    
    os.system(f"code {prefix + filepath}")

# Save figure
def save_fig(fig, filepath, prefix='images/', pad=0, h_pad=2, w_pad=2, pdf=False, dpi=350, tight_layout=True, show=False):
    """Saves a figure and copies it to the clipboard (for terminal use)."""
    filepath = clean_image_filepath(filepath)
    filetype = ".pdf" if pdf else ".png"
    if not filepath.endswith(filetype):
        filepath = filepath + filetype
    print_tmp(prefix + filepath)
    if tight_layout:
        fig.tight_layout(h_pad=h_pad, w_pad=w_pad, pad=pad)
    fig.savefig(prefix + filepath, dpi=dpi, bbox_inches='tight' if pad == 0 else None)
    if show:
        plt.show()
    else:
        plt.close(fig)
    copy(prefix + filepath)

def open_fig(filepath, prefix='images/'):
    filepath = clean_image_filepath(filepath)
    if not filepath.endswith(".png"):
        filepath = filepath + ".png"
    os.system(f"open {prefix + filepath}")

def email_fig(filepath, prefix='images/'):
    filepath = clean_image_filepath(filepath)
    if not filepath.endswith(".png"):
        filepath = filepath + ".png"
    os.system(f"cp {prefix + filepath} {EMAIL_PATH}")    

def email_df(filepath, prefix='data/'):
    filepath = clean_data_filepath(filepath)
    if not filepath.endswith(".csv"):
        filepath = filepath + ".csv"
    os.system(f"cp {prefix + filepath} {EMAIL_PATH}")    


# Miscelllaneous file types
def check_md5(filepath, filepath_md5):
    """Check the MD5 checksum of the file."""
    # Calculate the MD5 checksum of the file
    with open(filepath, 'rb') as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    calculated_md5 = file_hash.hexdigest()

    # Read the expected MD5 checksum from the .md5 file
    with open(filepath_md5, 'r') as f:
        expected_md5 = f.read().strip().split()[0]

    # Compare the calculated and expected MD5 checksums
    if calculated_md5 != expected_md5:
        raise ValueError(f"MD5 checksum does not match for {filepath}. Expected {expected_md5}, got {calculated_md5}.")
    else:
        print(f"Success: MD5 checksum matches for {filepath}.")

def gunzip(filepath):
    """Decompress a .gz file."""
    output_filepath = filepath.removesuffix(".gz")
    with gzip.open(filepath, 'rb') as f_in:
        with open(output_filepath, 'wb') as f_out:
            f_out.write(f_in.read())

## String conversion
def make_friendly(text):
    """Make a filename friendly for Linux."""
    if not isinstance(text, str):
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace spaces and common punctuation with underscores
    text = re.sub(r'[\s\-.,;:!?/\\()<>\[\]{}|]+', '_', text)
    
    # Remove any characters that aren't alphanumeric or underscore
    text = re.sub(r'[^a-z0-9_]+', '', text)
    
    # Replace multiple underscores with a single underscore
    text = re.sub(r'_+', '_', text)
    
    # Remove leading and trailing underscores
    text = text.strip('_')
    
    # Truncate to a reasonable length (e.g., 255 characters)
    text = text[:255]
    
    # Ensure the filename is not empty
    if not text:
        text = ""
    
    return text if text else "unnamed"

def crude_title(string):
    return string.replace("_", " ").title()

def save_float(data, filepath, prefix='data/', general=False):
    filepath = clean_data_filepath(filepath)
    if not filepath.endswith(".txt"):
        filepath += ".txt"
    if general:
        prefix = GENERAL_PATH
    with open(prefix + filepath, "w") as f:
        f.write(str(data))

def load_float(filepath, prefix='data/', general=False):
    filepath = clean_data_filepath(filepath)
    if not filepath.endswith(".txt"):
        filepath += ".txt"
    if general:
        prefix = GENERAL_PATH
    with open(prefix + filepath, "r") as f:
        return float(f.read())

def save_int(data, filepath, prefix='data/', general=False):
    filepath = clean_data_filepath(filepath)
    if not filepath.endswith(".txt"):
        filepath += ".txt"
    if general:
        prefix = GENERAL_PATH
    with open(prefix + filepath, "w") as f:
        f.write(str(data))

def load_int(filepath, prefix='data/', general=False):
    filepath = clean_data_filepath(filepath)
    if not filepath.endswith(".txt"):
        filepath += ".txt"
    if general:
        prefix = GENERAL_PATH
    with open(prefix + filepath, "r") as f:
        return int(f.read())

def save_list(data, filepath, prefix='data/', general=False):
    filepath = clean_data_filepath(filepath)
    if not filepath.endswith(".txt"):
        filepath += ".txt"
    if general:
        prefix = GENERAL_PATH
    with open(prefix + filepath, 'w') as f:
        for item in data:
            f.write(f"{item}\n")
    print(prefix + filepath)
    copy(prefix + filepath)

def load_list(filepath, prefix='data/', general=False):
    filepath = clean_data_filepath(filepath)
    if not filepath.endswith(".txt"):
        filepath += ".txt"
    if general:
        prefix = GENERAL_PATH
    with open(prefix + filepath, 'r') as f:
        data =  [line.strip() for line in f]
    if all([element.isdigit() for element in data]):
        data = [int(element) for element in data]
    return data 

def save_dict(data, filepath, prefix='data/', general=False):
    filepath = clean_data_filepath(filepath)
    if not filepath.endswith(".json"):
        filepath += ".json"
    if general:
        prefix = GENERAL_PATH
    with open(prefix + filepath, 'w') as f:
        json.dump(data, f)
    print(prefix + filepath)
    copy(prefix + filepath)

def load_dict(filepath, prefix='data/', general=False):
    filepath = clean_data_filepath(filepath)
    if not filepath.endswith(".json"):
        filepath += ".json"
    if general:
        prefix = GENERAL_PATH
    with open(prefix + filepath, 'r') as f:
        return json.load(f)

def save_string(data, filepath, prefix='data/', general=False):
    filepath = clean_data_filepath(filepath)
    if not filepath.endswith(".txt"):
        filepath += ".txt"
    if general:
        prefix = GENERAL_PATH
    with open(prefix + filepath, 'w') as f:
        f.write(data)

def load_string(filepath, prefix='data/', general=False):
    filepath = clean_data_filepath(filepath)
    if not filepath.endswith(".txt"):
        filepath += ".txt"
    if general:
        prefix = GENERAL_PATH
    with open(prefix + filepath, 'r') as f:
        return f.read()

def save_set(data, filepath, prefix='data/', general=False):
    filepath = clean_data_filepath(filepath)
    if not filepath.endswith(".json"):
        filepath += ".json"
    if general:
        prefix = GENERAL_PATH
    with open(prefix + filepath, 'w') as f:
        json.dump(list(data), f)
    print(prefix + filepath)
    copy(prefix + filepath)

def load_set(filepath, prefix='data/', general=False):
    filepath = clean_data_filepath(filepath)
    if not filepath.endswith(".json"):
        filepath += ".json"
    if general:
        prefix = GENERAL_PATH
    with open(prefix + filepath, 'r') as f:
        return set(json.load(f))

def open_series(series, name="open_series"):
    save_df(series.to_frame().reset_index(), name)
    open_df(name)

def save_df(data, filepath, index=False, prefix='data/', general=False, **kwargs):
    filepath = clean_data_filepath(filepath)
    if not filepath.endswith(".csv"):
        filepath += ".csv"
    if general:
        prefix = GENERAL_PATH
    data.to_csv(prefix + filepath, index=index, **kwargs)
    print(prefix + filepath)
    copy(prefix + filepath)

def open_df(filepath, prefix='data/', general=False):
    filepath = clean_data_filepath(filepath)
    if not filepath.endswith(".csv"):
        filepath += ".csv"
    if general:
        prefix = GENERAL_PATH
    os.system(f"open {prefix + filepath}")

def load_df(filepath, prefix='data/', low_memory=False, general=False, **kwargs):
    filepath = clean_data_filepath(filepath)
    if not filepath.endswith(".csv"):
        filepath += ".csv"
    if general:
        prefix = GENERAL_PATH

    df = pd.read_csv(prefix + filepath, low_memory=low_memory, **kwargs)

    return df

## Clipboard
def copy(data, index=False):
    if type(data) == pd.DataFrame or type(data) == pd.Series:
        data.to_clipboard(index=index)
    else:
        pyperclip.copy(str(data))

def copy_head(data, n=10, index=False):
    if type(data) == pd.DataFrame or type(data) == pd.Series:
        data.head(n).to_clipboard(index=index)
    elif type(data) == list:
        copy(data[:n])
    elif type(data) == dict:
        copy(data[:n])
    elif type(data) == set:
        copy(list(data)[:n])

## Set Comparison
class Compare:
    def __init__(self, data1, data2):
        if type(data1) != set:
            data1 = set(data1)
        if type(data2) != set:
            data2 = set(data2)
        self.data1 = data1
        self.data2 = data2

    def __str__(self):
        # Create table-like string representation
        output = [
            "Comparison Summary",
            "=" * 30,
            f"{'Set':12} | {'Count':>8} | {'%':>6}",
            "-" * 30,
            f"{'Left only':12} | {self.n_left:8,d} | {self.n_left/len(self.union):6.1%}",
            f"{'Right only':12} | {self.n_right:8,d} | {self.n_right/len(self.union):6.1%}", 
            f"{'Intersection':12} | {self.n_inner:8,d} | {self.n_inner/len(self.union):6.1%}",
            "-" * 30,
            f"{'Union':12} | {self.n_union:8,d} | {1.0:6.1%}"
        ]
        return "\n".join(output)

    def __repr__(self):
        return self.__str__()

    @property
    def same(self):
        return self.data1 == self.data2

    @property
    def n_data1(self):
        return len(self.data1)

    @property 
    def n_data2(self):
        return len(self.data2)
    
    @property
    def left(self):
        return self.data1 - self.data2

    @property
    def right(self):
        return self.data2 - self.data1

    @property
    def inner(self):
        return self.data1.intersection(self.data2)

    @property
    def union(self):
        return self.data1.union(self.data2)

    @property
    def outer(self):
        return self.data1.union(self.data2) - self.inner

    @property
    def n_left(self):
        return len(self.left)

    @property
    def n_right(self):
        return len(self.right)

    @property
    def n_inner(self):
        return len(self.inner)

    @property
    def n_union(self):
        return len(self.union)

    @property
    def n_outer(self):
        return len(self.outer)

    @property
    def left_in_right(self):
        """Check if data1 is a subset of data2"""
        return self.data1.issubset(self.data2)

    @property
    def right_in_left(self):
        """Check if data2 is a subset of data1"""
        return self.data2.issubset(self.data1)

    def to_dict(self):
        """Convert comparison results to dictionary"""
        return {
            'n_data1': self.n_data1,
            'n_data2': self.n_data2,
            'n_left': self.n_left,
            'n_right': self.n_right,
            'n_inner': self.n_inner,
            'n_union': self.n_union,
            'n_outer': self.n_outer,
            'jaccard': self.jaccard_similarity(),
            'dice': self.dice_coefficient(),
            'overlap': self.overlap_coefficient()
        }

def n_unique(data):
    return len(set(data))

def unique(data):
    return sorted(set(data))

def is_unique(data):
    return len(data) == n_unique(data)

## Git
def gitkeep(filepath):
    os.system(f"echo !{filepath} >> .gitignore")

def gitignore(filepath):
    os.system(f"echo {filepath} >> .gitignore")


## Pandas
def no_na(data):
    if isinstance(data, pd.Series):
        return data.isna().sum() == 0
    elif isinstance(data, pd.DataFrame):
        return data.isna().sum().sum() == 0
    elif isinstance(data, list):
        return all(no_na(item) for item in data)
    elif isinstance(data, np.ndarray):
        return np.isnan(data).sum() == 0
    else:
        raise ValueError(f"Unexpected type: {type(data)}")

def browse_df(df, i=0, n=20):
    if i < 0:
        return df.iloc[i:i-n:-1][::-1]
    else:
        return df.iloc[i:i+n]

def is_symmetric(matrix):
    return is_all(matrix, matrix.T, True)

def align(series_1, series_2):
    return (series_1 == series_2).all()

def is_all(series, value=True):
    return (series == value).all()

def one_per(series, group, feature=None):
    if feature is None:
        return is_all(series.groupby(group).size(), 1)
    return is_all(series.groupby(group)[feature].nunique(), 1)

def place_first(df, col):
    """Places the given column or list of columns first in the dataframe."""
    original_columns = df.columns.tolist()
    if isinstance(col, str):
        col = [col]
    # Rearrange columns: specified columns first, followed by the rest
    remaining_columns = [x for x in original_columns if x not in col]
    return df[col + remaining_columns]

def decompress_by_col(df, col, divider=","):
    dfo = pd.DataFrame()
    for i, row in df.iterrows():
        if "," in row[col]:
            for item in row[col].split(divider):
                new_row = row.copy()
                new_row[col] = item
                dfo = pd.concat([dfo, pd.DataFrame([new_row])])
        else:
            dfo = pd.concat([dfo, pd.DataFrame([row])])
    return dfo


## Statistics
def sd_to_mad(sd):
    return sd * norm.ppf(0.75)

def mad_to_sd(mad):
    return mad / norm.ppf(0.75)

def get_mad_normal(x):
    """
    Returns robust estimates of the median and standard deviation of the input distribution using median and MAD.
    """
    if isinstance(x, list):
        x = np.array(x)
        x = x[~np.isnan(x)]
    if type(x) == pd.Series:
        x = x.dropna()
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    sd = mad_to_sd(mad)
    return median, sd


def robust_zscore(df, col):
    mean, sd = get_mad_normal(df[col])
    return (df[col] - mean) / sd

def winsorize(df, measurement, winsor_size=3):
    lower_quantile = norm.cdf(-winsor_size)
    upper_quantile = norm.cdf(winsor_size)
    lower = df[measurement].quantile(lower_quantile)
    upper = df[measurement].quantile(upper_quantile)
    df[measurement] = df[measurement].clip(lower, upper)
    return df



## Genetics

def open_catalog(identifier):
	identifier_type = "score" if identifier[:3] == "PGS" else "publication"
	webpage = f"https://www.pgscatalog.org/{identifier_type}/{identifier}/"
	webbrowser.open(webpage)


def open_gwas(gene):
    webbrowser.open(f"https://www.ebi.ac.uk/gwas/genes/{gene}")



## Chrom-pos handling

def abstract_chrom(chrom):
    if chrom == "X":
        chrom = 23
    if chrom == "Y":
        chrom = 24
    return int(chrom)

def sort_chrom(chrom):
    if isinstance(chrom, str):
        return chrom
    elif isinstance(chrom, pd.Series) or isinstance(chrom, pd.Index):
        return sorted(set(chrom), key=abstract_chrom)
    elif isinstance(chrom, set):
        return sorted(chrom, key=abstract_chrom)
    elif isinstance(chrom, list):
        return sorted(chrom, key=abstract_chrom)
    else:
        raise ValueError(f"Unexpected type: {type(chrom)}")

def sort_chrom_pos(chrom_pos):
    if isinstance(chrom_pos, str):
        return chrom_pos
    elif isinstance(chrom_pos, pd.Series) or isinstance(chrom_pos, pd.Index):
        return sorted(set(chrom_pos), key=abstract_pos)
    elif isinstance(chrom_pos, set):
        return sorted(chrom_pos, key=abstract_pos)
    elif isinstance(chrom_pos, list):
        return sorted(chrom_pos, key=abstract_pos)
    else:
        raise ValueError(f"Unexpected type: {type(chrom_pos)}")

def abstract_pos(chrom_pos):
    chrom, pos = chrom_pos.split("_")
    if chrom == "X":
        chrom = 23
    if chrom == "Y":
        chrom = 24
    return int(chrom) * 1e9 + int(pos)

def make_chrom_pos_ref_alt_index(df):
    """Create a chrom_pos_ref_alt index for a dataframe."""
    df.index = get_cpra(df)
    return df

def make_chrom_pos_index(df):
    df.index = get_chrom_pos(df)
    return df

def sort_by_chrom_pos(df):
    """Warning: This will destroy the index of df and relies on no duplicate chrom-position pairs"""
    df = make_chrom_pos_index(df)
    df = df.loc[sort_chrom_pos(df.index)]
    df.reset_index(drop=True, inplace=True)
    return df

def select_on_chrom_pos(df, chrom_pos_set):
	return df[get_chrom_pos(df).isin(chrom_pos_set)].copy()

def join_on_chrom_pos_ref_alt(df1, df2, how="inner", lsuffix="", rsuffix="_2"):
	df1, df2 = df1.copy(), df2.copy()
	df1 = make_chrom_pos_ref_alt_index(df1)
	df2 = make_chrom_pos_ref_alt_index(df2)
	if how == "left" or how == "inner":
		df2.drop(["chrom", "pos", "ref", "alt"], axis=1, inplace=True)
	if how == "right":
		df1.drop(["chrom", "pos", "ref", "alt"], axis=1, inplace=True)
	df = df1.join(df2, how=how, lsuffix=lsuffix, rsuffix=rsuffix)
	df.reset_index(drop=True, inplace=True)
	return df

def join_on_chrom_pos(df1, df2, how="inner", lsuffix="_1", rsuffix="_2"):
	df1, df2 = df1.copy(), df2.copy()
	df1 = make_chrom_pos_index(df1)
	df2 = make_chrom_pos_index(df2)
	if how == "left" or how == "inner":
		df2.drop(["chrom", "pos"], axis=1, inplace=True)
	if how == "right":
		df1.drop(["chrom", "pos"], axis=1, inplace=True)
	df = df1.join(df2, how=how, lsuffix=lsuffix, rsuffix=rsuffix)
	df.reset_index(drop=True, inplace=True)
	return df

def make_chrom_pos_col(df):
	df["chrom_pos"] = get_chrom_pos(df)
	return df

def make_cpr_col(df):
	df["cpr"] = get_cpr(df)
	return df

def make_cpra_col(df):
	df["cpra"] = get_cpra(df)
	return df

def get_chrom_pos(df):
	return df.chrom.astype(str) + "_" + df.pos.astype(str)

def get_cpr(df):
	return df.chrom.astype(str) + "_" + df.pos.astype(str) + "_" + df.ref.astype(str)

def get_cpra(df):
	return df.chrom.astype(str) + "_" + df.pos.astype(str) + "_" + df.ref.astype(str) + "_" + df.alt.astype(str)

def str_chrom(df):
    if "chrom" in df.columns:
        df["chrom"] = df["chrom"].astype(str)
    if "CHROM" in df.columns:
        df["CHROM"] = df["CHROM"].astype(str)
    return df

def strip_chr(df):
    if "chrom" in df.columns:
        df["chrom"] = df["chrom"].map(lambda x: x[3:] if "chr" in x else x)
    if "CHROM" in df.columns:
        df["CHROM"] = df["CHROM"].map(lambda x: x[3:] if "chr" in x else x)
    return df


## VCF handling

def get_vcf_sample_names(filepath):
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#CHROM'):
                standard_cols = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT']
                standard_cols_w_hash = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT']
                assert line.strip().split('\t')[:9] == standard_cols_w_hash, "First 9 VCF columns do not match expected standard columns"
                cols = line.strip().split('\t')
                sample_names = cols[len(standard_cols):]
                return sample_names
    return None


def get_vcf_columns(filepath):
    vcf_sample_names = get_vcf_sample_names(filepath)
    if vcf_sample_names is None:
        return None
    return ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'] + vcf_sample_names


def load_vcf(filepath):
    """
    Loads a VCF file (.vcf) into a pandas DataFrame. 
    Run time: about 2-4 minutes depending on the size of the file.

    It should handle any number of samples. 
    """
    vcf_sample_names = get_vcf_sample_names(filepath)
    vcf_columns = get_vcf_columns(filepath)
    df = pd.read_csv(filepath, sep='\t', comment='#', header=None, low_memory=False)
    df.columns = vcf_columns

    # Drop rows with no FORMAT
    if df.FORMAT.isna().sum() > 0:
        assert df[df.FORMAT.isna()][vcf_sample_names].isna().all()
        df.dropna(subset=["FORMAT"], inplace=True)

    # Drop these columns if they are unique
    if is_all(df["INFO"], "."):
        df.drop(columns=["INFO"], inplace=True)

    if is_all(df["ID"], "."):
        df.drop(columns=["ID"], inplace=True)

    if is_all(df["FILTER"], "PASS"):
        df.drop(columns=["FILTER"], inplace=True)

    df.rename(columns={ 'CHROM': 'chrom', 'POS': 'pos', "ID": "rsid", 'REF': 'ref', 'ALT': 'alt', 'QUAL': 'qual', 
    'INFO': 'information', 'FILTER': 'filter_status'}, inplace=True)

    dfk = pd.DataFrame()

    # Get unique format types and their frequencies
    format_types = list(set(df.FORMAT))

    # Create format mappings programmatically
    format_mappings = {}
    for format_type in format_types:
        format_mappings[format_type] = format_type.split(':')
    
    # Process each format type
    for format_type in format_types:
        # Get rows with this format
        df_temp = df[df.FORMAT == format_type]
        
        if len(df_temp) == 0:
            continue
            
        # Get corresponding format columns
        format_cols = format_mappings[format_type]
        
        # Process each sample
        for sample_name in vcf_sample_names:
            # Split sample data into columns
            sample_expanded = df_temp[sample_name].str.split(':', expand=True)
            if sample_expanded.shape[1] > 0:  # Only process if we have data
                sample_expanded = sample_expanded.iloc[:, :len(format_cols)]
                sample_expanded.columns = [f"{sample_name}_{col}" for col in format_cols]
                
                # Convert GQ and DP columns to float, replacing "." with np.nan
                for col in format_cols:
                    if col in ['GQ', 'DP']:
                        col_name = f"{sample_name}_{col}"
                        sample_expanded.loc[sample_expanded[col_name] == ".", col_name] = np.nan
                        sample_expanded[col_name] = sample_expanded[col_name].astype(float)
                
                df_temp = df_temp.join(sample_expanded)
                df_temp.drop(columns=[sample_name], inplace=True)
                
        # Update original dataframe
        dfk = pd.concat([dfk, df_temp])

    for sample_name in vcf_sample_names:
        depths = dfk[f'{sample_name}_AD'].str.split(',', expand=True).replace(".", np.nan).astype(float)

        dfk[f'{sample_name}_ref_depth'] = depths[0]
        if depths.shape[1] > 2:
            for i in range(1, depths.shape[1]):
                dfk[f'{sample_name}_alt_depth_{i}'] = depths[i]
        else:
            dfk[f'{sample_name}_alt_depth'] = depths[1]


    dfk.drop(columns=["FORMAT"] + [f"{sample_name}_AD" for sample_name in vcf_sample_names], inplace=True)
    dfk = strip_chr(dfk)
    return dfk




