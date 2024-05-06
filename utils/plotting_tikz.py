import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patheffects import withStroke
from scipy.stats import t
from scipy.ndimage import uniform_filter1d
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",  # or any other engine you want to use
    "font.size": 11,  # control font sizes of different elements
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "font.family": "serif",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pgf.preamble": r"""
\usepackage[T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[russian]{babel}
"""
})
sample = 32
colors = np.vstack((
    plt.get_cmap("YlOrBr", sample)(np.linspace(0, 1, sample))[::-1],
    plt.get_cmap("YlGn", sample)(np.linspace(0, 1, sample)),
)
)  # [::-1]

colormap = LinearSegmentedColormap.from_list("green_to_red", colors)
colormap = 'RdYlGn'

PICS_DIRECTORY = 'pics'


def plot_actions_batch(agent_actions_history, title='', mode='mean', num=0):
    data = agent_actions_history
    combined_data = []
    yticklabels = []
    indicator_labels = {
        'percent_to_buy': r'$B',
        'percent_to_sale': r'$sell',
        'percent_to_use': '',
        'prices': r'$p'
    }
    n_firms, n_branches = data[0]['percent_to_buy'].shape[1:]
    for key in ['percent_to_buy', 'percent_to_use', 'percent_to_sale', 'prices', ]:
        if mode == 'mean':
            key_data = np.array([entry[key].mean(axis=0).flatten() for entry in data])
        else:
            key_data = np.array([np.median(entry[key], axis=0).flatten() for entry in data])
        combined_data.append(key_data)
        if key == 'percent_to_buy':
            for firm in range(n_firms):
                for branch in range(n_branches):
                    yticklabels.append(indicator_labels[key] + "_{" + f"{branch + 1}" + ", " + f"({firm + 1})" + "}$")
        elif key == 'percent_to_use':
            if key_data.shape[1] == n_branches:
                yticklabels += ["$\Pi_" + f"{i + 1}$" for i in range(key_data.shape[1])]
            else:
                yticklabels += ["$\Pi_" + f"{i + 1}$" for i in range(key_data.shape[1] // 2)]
                yticklabels += ["$I_" + f"{i + 1}$" for i in range(key_data.shape[1] // 2)]
        else:
            yticklabels += [indicator_labels[key] + "_{" + f"{i + 1}" + "}$" for i in range(key_data.shape[1])]
    fig = plt.figure(figsize=(6, 4))
    fig.subplots_adjust(left=0.075)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(bottom=0.1)
    combined_data = np.hstack(combined_data).T
    sns.heatmap(combined_data,
                annot=False,
                cmap=colormap,
                cbar=False,
                yticklabels=yticklabels)
    plt.hlines(np.arange(1, len(yticklabels)),
               *plt.xlim(),
               colors='black',
               linestyles='solid',
               linewidth=1,
               )
    plt.xlabel("Шаг")
    plt.ylabel("")
    plt.yticks(rotation=0)
    # plt.gca().set_yticklabels(plt.gca().get_yticklabels())
    # plt.gca().tick_params(axis='y', which='major')
    plt.savefig(f'{PICS_DIRECTORY}/actions_{title}_{num}.pgf')


def plot_volumes_batch(env_history, confidence=0.75, alpha=0.5, window_size=5, num=0):
    volumes = np.stack([x['volume_matrix'] for x in env_history])  # .T
    reserves = np.stack([x['reserves'] for x in env_history]).transpose(0, 2, 1, 3)
    total = (volumes + reserves).sum(axis=2)
    n = volumes.shape[0]
    common = t.ppf((1 + confidence) / 2., n - 1) / np.sqrt(n)
    total_mean = total.mean(axis=1)
    total_h = total.std(axis=1) * common
    plt.figure(figsize=(6, 3.5))
    for commodity in range(total_h.shape[1]):
        data_ma = uniform_filter1d(total_mean[:, commodity], size=window_size, axis=0, mode='nearest')
        plt.plot(data_ma,
                 label=f'Товар {commodity + 1}',
                 path_effects=[withStroke(linewidth=2, foreground='black')],
                 )
        plt.fill_between(
            range(total.shape[0]),
            total_mean[:, commodity] - total_h[:, commodity],
            total_mean[:, commodity] + total_h[:, commodity],
            alpha=alpha,
        )
    plt.xlabel('Шагов')
    plt.ylabel('Общий объём')
    plt.legend()
    plt.savefig(f'{PICS_DIRECTORY}/volumes_{num}.pgf')


def plot_environment_batch(env_history, confidence=0.8, alpha=0.5, window_size=5, num=0):
    data = env_history
    num_firms, n_branches = data[0]['price_matrix'].shape[1:]

    prices = np.stack([x['price_matrix'] for x in data]).transpose(2, 3, 1, 0)
    volumes = np.stack([x['volume_matrix'] for x in data]).transpose(2, 3, 1, 0)
    reserves = np.stack([x['reserves'] for x in data]).transpose(2, 3, 1, 0)
    finances = np.stack([x['finance'] for x in data]).transpose(0, 2, 1)
    finances = uniform_filter1d(finances, size=window_size, axis=0, mode='nearest')
    periods = np.arange(len(data))

    n = finances.shape[2]
    common = t.ppf((1 + confidence) / 2., n - 1) / np.sqrt(n)

    finances_mean = finances.mean(axis=2)
    finances_h = finances.std(axis=2) * common

    fig, axes = plt.subplots(nrows=n_branches, ncols=3, figsize=(6, n_branches * 2.5), sharex='col',
                             gridspec_kw={'wspace': 0.2, 'hspace': 0.1}
                             )
    fig.subplots_adjust(left=0.08)
    for ax in axes.flatten():
        ax.tick_params(axis='y', which='both', pad=0)
    colors = plt.cm.get_cmap('Set1').colors

    def plot_data(ax, data, item_index, title, logscale=False):
        data_mean = data.mean(axis=2)
        data_h = data.std(axis=2) * common
        for firm_index in range(num_firms):
            data_ma = uniform_filter1d(data_mean[firm_index, item_index], size=window_size, axis=0, mode='nearest')
            ax.plot(periods, data_ma,
                    color=colors[firm_index % len(colors)],
                    label=f'Фирма {firm_index + 1}',
                    # linewidth=2,
                    path_effects=[withStroke(linewidth=2, foreground='black')]
                    )
            ax.fill_between(
                periods,
                data_mean[firm_index, item_index] - data_h[firm_index, item_index],
                data_mean[firm_index, item_index] + data_h[firm_index, item_index],
                alpha=alpha,
                label=f'Фирма {firm_index + 1}',
                color=colors[firm_index % len(colors)],

            )
        if item_index == 0:
            ax.set_title(title)
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        if logscale:
            ax.set_yscale('log')

    def setup_axis(ax, item_index, data, label, is_last_row=False, logscale=False):
        plot_data(ax, data, item_index, label, logscale)
        if is_last_row:
            ax.set_xlabel('Период')

    for item_index in range(n_branches):
        axes[item_index, 0].set_ylabel(f'Товар {item_index + 1}')
        setup_axis(axes[item_index, 0], item_index, prices, 'Цены', item_index == n_branches - 1)
        setup_axis(axes[item_index, 1], item_index, volumes, 'Рыночные объёмы', item_index == n_branches - 1)
        setup_axis(axes[item_index, 2], item_index, reserves, 'Резервы', item_index == n_branches - 1)

    handles, labels = [], []
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    # fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(1.1, 1), title='Легенда')
    plt.subplots_adjust(right=1)
    plt.savefig(f'{PICS_DIRECTORY}/timedata_{num}.pgf')
    ncols = 1 + ('limits' in data[0].keys())
    fig, ax = plt.subplots(1, ncols, figsize=(3 * ncols, 3.5))
    ax = np.atleast_1d(ax)
    for i in range(finances_mean.shape[1]):
        data_ma = uniform_filter1d(finances_mean[:, i], size=window_size, axis=0, mode='nearest')

        ax[0].plot(periods,
                   data_ma,
                   # linewidth=2,
                   path_effects=[withStroke(linewidth=2, foreground='black')],
                   label=labels[i],
                   color=colors[i]
                   )
        ax[0].fill_between(
            periods,
            finances_mean[:, i] - finances_h[:, i],
            finances_mean[:, i] + finances_h[:, i],
            color=colors[i],
            label=None,
            alpha=alpha
        )
    ax[0].set_title('Финансы')
    ax[0].set_xlabel('Период')
    ax[0].set_ylabel('Финансы')
    # ax[0].legend()
    ax[0].grid(True)
    if ncols == 2:
        limits = np.stack([x['limits'] for x in data]).mean(axis=2).transpose(1, 0, 2)  #
        limits_h = np.stack([x['limits'] for x in data]).std(axis=2).transpose(1, 0, 2) * common

        for i in range(limits.shape[0]):
            ax[1].plot(periods, limits[i].flatten(),
                       label=labels[i],
                       color=colors[i],
                       path_effects=[withStroke(linewidth=2, foreground='black')]
                       )
            ax[1].fill_between(
                periods,
                (limits[i] - limits_h[i]).flatten(),
                (limits[i] + limits_h[i]).flatten(),
                label=None,
                alpha=alpha,
                color=colors[i])

        ax[1].set_title('Лимиты')
        ax[1].set_xlabel('Период')
        ax[1].set_ylabel('Лимиты')
        # ax[1].legend()
        ax[1].grid(True)
    plt.savefig(f'{PICS_DIRECTORY}/timedata2_{num}.pgf')


def plot_loss_batch(df_list, num=0):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 4))  # Изменим размер и расположение графиков
    df = pd.concat(df_list).reset_index()
    for firm_id, group in df.groupby("firm_id"):
        color = plt.cm.get_cmap('Set1').colors[firm_id]

        ax[0, 0].plot(
            group["episode"],
            group["actor_loss"],
            label=f"Firm {firm_id}",
            path_effects=[withStroke(linewidth=2, foreground='black')],
            color=color,
        )
        ax[0, 1].plot(
            group["episode"],
            group["critic_loss"],
            label=f"Firm {firm_id}",
            path_effects=[withStroke(linewidth=2, foreground='black')],
            color=color,
        )
        ax[1, 0].plot(
            group["episode"],
            group["reward"],
            label=f"Firm {firm_id}",
            path_effects=[withStroke(linewidth=2, foreground='black')],
            color=color,
        )

        ax[1, 1].plot(
            group["episode"],
            group["entropy_loss"],
            label=f"Фирма {firm_id}",
            path_effects=[withStroke(linewidth=2, foreground='black')],
            color=color,
        )

    ax[0, 0].set_xlabel("Эпоха")
    ax[0, 0].set_title("Actor Loss")
    ax[0, 0].grid(True)

    ax[0, 1].set_xlabel("Эпоха")
    ax[0, 1].set_title("Critic Loss")
    ax[0, 1].grid(True)

    ax[1, 0].set_xlabel("Эпоха")
    ax[1, 0].set_title("Средняя награда")
    # ax[1, 0].legend()
    ax[1, 0].grid(True)
    ax[1, 0].axhline(y=0, color="grey", linestyle="--", label="Zero Profit")
    #
    ax[1, 1].set_xlabel("Эпоха")
    ax[1, 1].set_title("Entropy Loss")
    # ax[1, 1].legend()
    ax[1, 1].grid(True)
    #
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{PICS_DIRECTORY}/loss_{num}.pgf')


if __name__ == '__main__':
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 2))

    axes[0].imshow(gradient, aspect='auto', cmap=colormap)
    axes[0].set_title('Custom Green-Yellow-Red')
    axes[0].axis('off')

    axes[1].imshow(gradient, aspect='auto', cmap='RdYlGn')
    axes[1].set_title('Standard RdYlGn')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
