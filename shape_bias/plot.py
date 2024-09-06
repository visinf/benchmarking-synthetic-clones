import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
import os
import json 
import glob
from palettable.colorbrewer.qualitative import Dark2_8

PLOTTING_EDGE_COLOR = (0.3, 0.3, 0.3, 0.3)
PLOTTING_EDGE_WIDTH = 0.02
cat_dict = {
    'airplane': 0, 'bear': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'car': 6,
    'cat': 7, 'chair': 8, 'clock': 9, 'dog': 10, 'elephant': 11, 'keyboard': 12, 'knife': 13, 
    'oven': 14, 'truck': 15
}

def get_short_imagename(imagename):
    """Return image-specific suffix of imagename.

    This excludes a possible experiment-specific prefix,
    such as 0001_... for trial #1.
    """

    splits = imagename.split("_")
    if len(splits) > 1:
        name = splits[-2:]
        if name[0].startswith("n0"):
            # ImageNet image: keep n0... prefix
            name = name[0] + "_" + name[1]
        else:
            name = name[1]
    else:
        name = splits[0]
    return name


class Analysis(ABC):
    figsize = (7, 6)

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _check_dataframe(df):
        assert len(df) > 0, "empty dataframe"

    @abstractmethod
    def analysis(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_result_df(self, *args, **kwars):
        pass

    @property
    @abstractmethod
    def num_input_models(self):
        """Return number of input data frames for analysis.

        E.g. if analysis compares two observers/models, this
        number will be 2.
        """
        pass


class ShapeBias(Analysis):
    """Reference: Geirhos et al. ICLR 2019
    https://openreview.net/pdf?id=Bygh9j09KX
    """
    num_input_models = 1

    def __init__(self):
        super().__init__()
        self.plotting_name = "shape-bias"


    def analysis(self, df):

        self._check_dataframe(df)

        df = df.copy()
        df["correct_texture"] = df["imagename"].apply(self.get_texture_category)
        df["correct_shape"] = df["category"]

        # remove those rows where shape = texture, i.e. no cue conflict present
        df2 = df.loc[df.correct_shape != df.correct_texture]
        fraction_correct_shape = len(df2.loc[df2.object_response == df2.correct_shape]) / len(df)
        fraction_correct_texture = len(df2.loc[df2.object_response == df2.correct_texture]) / len(df)
        shape_bias = fraction_correct_shape / (fraction_correct_shape + fraction_correct_texture)

        result_dict = {"fraction-correct-shape": fraction_correct_shape,
                       "fraction-correct-texture": fraction_correct_texture,
                       "shape-bias": shape_bias}
        return result_dict


    def get_result_df(self):
        pass

    def get_texture_category(self, imagename):
        """Return texture category from imagename.

        e.g. 'XXX_dog10-bird2.png' -> 'bird
        '"""
        assert type(imagename) is str

        # remove unneccessary words
        a = imagename.split("_")[-1]
        # remove .png etc.
        b = a.split(".")[0]
        # get texture category (last word)
        c = b.split("-")[-1]
        # remove number, e.g. 'bird2' -> 'bird'
        d = ''.join([i for i in c if not i.isdigit()])
        return d


def read_all_csv_files_from_directory(dir_path):
    assert os.path.exists(dir_path)
    assert os.path.isdir(dir_path)

    df = pd.DataFrame()
    for f in sorted(os.listdir(dir_path)):
        if f.endswith(".csv"):
            df2 = pd.read_csv(os.path.join(dir_path, f))
            df2.columns = [c.lower() for c in df2.columns]
            df = pd.concat([df, df2])
    return df

def get_experimental_data(path, print_name=False):
    """Read all available data for an experiment."""


    df = read_all_csv_files_from_directory(path)
    df.condition = df.condition.astype(str)

    df = df.copy()
    df["image_id"] = df["imagename"].apply(get_short_imagename)

    return df



def plot_shape_bias_matrixplot(result_dir='/visinf/home/ksingh/syn-rep-learn/shape_bias/results',
                               order_by='humans'):
    os.makedirs(result_dir, exist_ok=True)
    analysis = ShapeBias()
    df = get_experimental_data('/visinf/home/ksingh/syn-rep-learn/shape_bias/data/cue-conflict')

    fontsize = 25
    ticklength = 10
    markersize = 250

    classes = df["category"].unique()
    num_classes = len(classes)

    # plot setup
    fig = plt.figure(1, figsize=(14, 14), dpi=600.)
    ax = plt.gca()

    ax.set_xlim([0, 1])
    ax.set_ylim([-.5, num_classes - 0.5])

    # secondary reversed x axis
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: 1 - x, lambda x: 1 - x))

    # labels, ticks
    plt.tick_params(axis='y',
                    which='both',
                    left=False,
                    right=False,
                    labelleft=False)
    ax.set_ylabel("Shape categories", labelpad=60, fontsize=fontsize)
    ax.set_xlabel("Fraction of 'texture' decisions", fontsize=fontsize, labelpad=25)
    ax_top.set_xlabel("Fraction of 'shape' decisions", fontsize=fontsize, labelpad=25)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax_top.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.get_xaxis().set_ticks(np.arange(0, 1.1, 0.1))
    ax_top.set_ticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='both', which='major', labelsize=fontsize, length=ticklength)
    ax_top.tick_params(axis='both', which='major', labelsize=fontsize, length=ticklength)

    # arrows on x axes
    plt.arrow(x=0, y=-1.75, dx=1, dy=0, fc='black',
              head_width=0.4, head_length=0.03, clip_on=False,
              length_includes_head=True, overhang=0.5)
    plt.arrow(x=1, y=num_classes + 0.75, dx=-1, dy=0, fc='black',
              head_width=0.4, head_length=0.03, clip_on=False,
              length_includes_head=True, overhang=0.5)
    path =  '/visinf/home/ksingh/syn-rep-learn/shape_bias/results/'
    fns = [
            f'{path}/resnet50.json',
            f'{path}/syn_clone.json',
            f'{path}/vit-b.json',
            f'{path}/scaling_imagenet_sup.json',
            f'{path}/dino.json',
            f'{path}/synclr.json',
            f'{path}/CLIP.json', 
            f'{path}/scaling_clip.json', 
           ]
    result_dict = []
    for fn in fns:
        with open(fn, 'r') as f:
            data_dict = json.load(f)
            result_dict.append(
                {
                    'model_name': data_dict['model_name'],
                    'shape_bias': data_dict['shape_bias'],
                    'shape_bias_per_cat': data_dict['shape_bias_per_cat'],
                }
            )
    class_avgs = []
    for cl in cat_dict.keys():
        class_avgs.append(1-analysis.analysis(df.query(f"category=='{cl}'"))['shape-bias'])
    sorted_indices = np.argsort(class_avgs)
    classes = np.asarray(list(cat_dict.keys()))[sorted_indices]
    # icon placement is calculated in axis coordinates
    WIDTH = 1 / num_classes  #
    XPOS = -1.25 * WIDTH  # placement left of yaxis (-WIDTH) plus some spacing (-.25*WIDTH)
    YPOS = -0.5
    HEIGHT = 1
    MARGINX = 1 / 10 * WIDTH  # vertical whitespace between icons
    MARGINY = 1 / 10 * HEIGHT  # horizontal whitespace between icons

    left = XPOS + MARGINX
    right = XPOS + WIDTH - MARGINX

    for i in range(num_classes):
        bottom = i + MARGINY + YPOS
        top = (i + 1) - MARGINY + YPOS
        iconpath = os.path.join('/visinf/home/ksingh/syn-rep-learn/shape_bias/icons', "{}.png".format(classes[i]))
        plt.imshow(plt.imread(iconpath), extent=[left, right, bottom, top], aspect='auto', clip_on=False)

    # plot horizontal intersection lines
    for i in range(num_classes - 1):
        plt.plot([0, 1], [i + .5, i + .5], c='gray', linestyle='dotted', alpha=0.4)


    class_avgs = []
    for cl in classes: 
        class_avgs.append(1-analysis.analysis(df.query(f"category=='{cl}'"))['shape-bias'])
        result_df = analysis.analysis(df)
    avg = 1 - result_df['shape-bias']
    ax.plot([avg, avg], [-1, num_classes], color='black')
    ax.scatter(class_avgs, classes,
                color='black',
                marker='d',
                label='human',
                s=markersize,
                clip_on=False,
                edgecolors=PLOTTING_EDGE_COLOR,
                linewidths=PLOTTING_EDGE_WIDTH,
                zorder=3)
    colors = Dark2_8.hex_colors
    #  colors = ['#21C5F4', '#FFC61A', '#ED9122', '#CCCCFF', '#8CE1F9']
    colors = ['#FF3333', '#FF3333', '#FFC61A','#FFC61A', '#ED9122', '#CBCBFC', '#8CE1F9', '#8CE1F9']
    labels = ['Resnet50', 'Resnet-50 (Syn.)', 'ViT-B', 'ViT-B (Syn.)', 'DINOv2', 'SimCLR (Syn.)', 'CLIP', 'CLIP (Syn.)']
    markers = ['s', 's', 's', 's', '^', '^', 'p', 'p']
    for i in range(0, len(result_dict)):
        class_avgs = []
        for cl in classes: 
            if cl not in result_dict[i]['shape_bias_per_cat'].keys():
                cl_avg = 1
            else:
                cl_avg = 1-result_dict[i]['shape_bias_per_cat'][cl]
            class_avgs.append(cl_avg)
            # print(f'cl:{cl}, val: {class_avgs[-1]}')
        avg = 1 - result_dict[i]['shape_bias']
        print(f"{__file__}, i: {i}")
        ax.plot([avg, avg], [-1, num_classes], color=colors[i])
        if i%2==0:
            ax.scatter(class_avgs, classes,
                        color=colors[i],
                        label=labels[i],
                        marker=markers[i],
                        s=markersize,
                        clip_on=False,
                        facecolors='none',
                        edgecolors=colors[i],
                        linewidths=PLOTTING_EDGE_WIDTH+1,
                        zorder=3)
        else:
            ax.scatter(class_avgs, classes,
                        color=colors[i],
                        label=labels[i],
                        marker=markers[i],
                        s=markersize,
                        clip_on=False,
                        edgecolors=PLOTTING_EDGE_COLOR,
                        linewidths=PLOTTING_EDGE_WIDTH,
                        zorder=3)

    lgd = plt.legend(loc='lower center', bbox_to_anchor=(0.5,-0.18), labelspacing=1.2, ncols=9, borderpad=0.8)
    # text = ax.text(-0.2,1.05, "Aribitrary text", transform=ax.transAxes)
    # plt.legend(loc=(1, -0.5), labelspacing=1.2, ncols=8)
    figure_path = os.path.join(result_dir, f"shape-bias_matrixplot.pdf")
    fig.savefig(figure_path,bbox_extra_artists=(lgd,ax), bbox_inches='tight')
    plt.close()

    # plot average shapebias + scatter points
    # for dmaker in decision_maker_fun(df):
    #     df_selection = df.loc[(df["subj"].isin(dmaker.decision_makers))]
    #     result_df = analysis.analysis(df=df_selection)
    #     avg = 1 - result_df['shape-bias']
    #     ax.plot([avg, avg], [-1, num_classes], color=dmaker.color)
    #     class_avgs = []
    #     for cl in classes:
    #         df_class_selection = df_selection.query("category == '{}'".format(cl))
    #         class_avgs.append(1 - analysis.analysis(df=df_class_selection)['shape-bias'])

    #     ax.scatter(class_avgs, classes,
    #                color=dmaker.color,
    #                marker=dmaker.marker,
    #                label=dmaker.plotting_name,
    #                s=markersize,
    #                clip_on=False,
    #                edgecolors=PLOTTING_EDGE_COLOR,
    #                linewidths=PLOTTING_EDGE_WIDTH,
    #                zorder=3)

    # figure_path = os.path.join(result_dir, f"{ds.name}_shape-bias_matrixplot.pdf")
    # fig.savefig(figure_path)
    # plt.close()

if __name__ == '__main__':
    plot_shape_bias_matrixplot()
