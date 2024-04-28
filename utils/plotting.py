import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_actions(agent_actions_history, title=''):
    data = agent_actions_history
    combined_data = []
    yticklabels = []
    indicator_labels = {
        'percent_to_buy': 'Покупка %',
        'percent_to_sale': 'Продажа %',
        'percent_to_use': '',
        'prices': 'Цена'
    }
    n_firms, n_branches = data[0]['percent_to_buy'].shape
    for key in ['percent_to_buy', 'percent_to_use', 'percent_to_sale', 'prices']:
        key_data = np.array([entry[key].flatten() for entry in data])
        combined_data.append(key_data)
        if key == 'percent_to_buy':
            for firm in range(n_firms):
                for branch in range(n_branches):
                    yticklabels.append(f"{indicator_labels[key]} (Т_{branch + 1}, Ф_{firm + 1})")
        elif key == 'percent_to_use':
            if key_data.shape[1] == n_branches:
                yticklabels += [f"Производство %  (Т_{i + 1})" for i in range(key_data.shape[1])]
            else:
                yticklabels += [f"Производство % (Т_{i + 1})" for i in range(key_data.shape[1] // 2)]
                yticklabels += [f"Инвестиции   %  (Т_{i + 1})" for i in range(key_data.shape[1] // 2)]
        else:
            yticklabels += [f"{indicator_labels[key]} (Т_{i + 1})" for i in range(key_data.shape[1])]

    combined_data = np.hstack(combined_data).T
    plt.figure(figsize=(16, 8))
    sns.heatmap(combined_data, annot=False, cmap='RdYlGn', cbar=True, yticklabels=yticklabels)
    plt.hlines(np.arange(1, len(yticklabels)), *plt.xlim(), colors='black', linestyles='solid', linewidth=1)

    plt.title(title)
    plt.xlabel("Period")
    plt.ylabel("Indicator")
    plt.yticks(rotation=0)
    plt.gca().set_yticklabels(plt.gca().get_yticklabels(), ha='left')
    plt.gca().tick_params(axis='y', which='major', pad=120)
    plt.show()


def plot_volumes(env_history):
    volumes = np.stack([x['volume_matrix'] for x in env_history]).T
    reserves = np.stack([x['reserves'] for x in env_history]).T
    total = (volumes + reserves).sum(axis=1).T
    plt.plot(total)
    plt.xlabel('Время')
    plt.ylabel('Общий объём')
    plt.title('Объём')
    plt.legend([f'Товар_{i}' for i in range(total.shape[1])])
    plt.show()

def plot_environment(env_history):
    data = env_history
    num_firms, n_branches = data[0]['price_matrix'].shape

    prices = np.stack([x['price_matrix'] for x in data]).T.transpose(1, 0, 2)
    volumes = np.stack([x['volume_matrix'] for x in data]).T.transpose(1, 0, 2)
    reserves = np.stack([x['reserves'] for x in data]).T.transpose(1, 0, 2)
    finances = np.stack([x['finance'] for x in data]).T

    fig, axes = plt.subplots(nrows=n_branches, ncols=3, figsize=(20, n_branches * 4), sharex='col',
                             gridspec_kw={'hspace': 0.1, 'wspace': 0.3})
    periods = np.arange(len(data))
    colors = plt.cm.get_cmap('Set1').colors

    def plot_data(ax, data, item_index, title, logscale=False):
        for firm_index in range(num_firms):
            ax.plot(periods, data[firm_index][item_index],
                    color=colors[firm_index % len(colors)],
                    label=f'Фирма {firm_index + 1}')
        if item_index == 0:
            ax.set_title(title)
        ax.grid(True)
        if logscale:
            ax.set_yscale('log')

    def setup_axis(ax, item_index, data, label, is_last_row=False, logscale=False):
        plot_data(ax, data, item_index, label, logscale)
        if is_last_row:
            ax.set_xlabel('Период')

    for item_index in range(n_branches):
        axes[item_index, 0].set_ylabel(f'Товар {item_index + 1}')
        setup_axis(axes[item_index, 0], item_index, prices, 'Цены', item_index == n_branches - 1)
        setup_axis(axes[item_index, 1], item_index, volumes, 'Объёмы', item_index == n_branches - 1)
        setup_axis(axes[item_index, 2], item_index, reserves, 'Резервы', item_index == n_branches - 1)

    handles, labels = [], []
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1), title='Легенда')
    plt.subplots_adjust(right=0.98)
    plt.show()
    ncols = 1 + ('limits' in data[0].keys())
    fig, ax = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    ax = np.atleast_1d(ax)

    for i in range(finances.shape[0]):
        ax[0].plot(periods, finances[i], label=labels[i], color=colors[i])
    ax[0].set_title('Финансы')
    ax[0].set_xlabel('Период')
    ax[0].set_ylabel('Финансы')
    ax[0].legend()
    ax[0].grid(True)
    if ncols == 2:
        limits = np.stack([x['limits'] for x in data]).T
        for i in range(limits.shape[0]):
            ax[1].plot(periods, limits[i], label=labels[i], color=colors[i])
        ax[1].set_title('Лимиты')
        ax[1].set_xlabel('Период')
        ax[1].set_ylabel('Лимиты')
        ax[1].legend()
        ax[1].grid(True)
    plt.show()
