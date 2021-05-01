import seaborn as sns
import matplotlib.pyplot as plt

from patterns import PatternMatrix, PatternType


def main():
    pattern_type = 0
    pattern_matrix = PatternMatrix(pattern_type).matrix
    fig, ax = plt.subplots(ncols=2, nrows=1, sharex=False, sharey=True,
                           gridspec_kw={'wspace': 0.1, 'hspace': 0.3})
    g = sns.heatmap(pattern_matrix[:, 0].reshape((8, 8)), cbar=False,
                    ax=ax[0], cbar_kws={"shrink": 0.4}, center=0.0)
    # ax[0].title.set_text('Pattern Matrix $A$')
    ax[0].title.set_fontsize(10)
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].yaxis.set_visible(False)
    ax[0].xaxis.set_visible(False)
    g = sns.heatmap(pattern_matrix[:, 1].reshape((8, 8)), cbar=False,
                    ax=ax[1], cbar_kws={"shrink": 0.4}, center=0.0)
    # ax[1].title.set_text('Pattern Matrix $A$')
    ax[1].title.set_fontsize(10)
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].yaxis.set_visible(False)
    ax[1].xaxis.set_visible(False)
    output_path = f'../results/plots/pattern_and_distractor_matrices_pattern_type_{pattern_type}.png'
    fig.savefig(fname=output_path, orientation='landscape',
                dpi=75, bbox_inches='tight', pad_inches=0)
    plt.close(fig=fig)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print(e)
