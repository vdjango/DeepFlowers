import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1 import host_subplot


# matplotlib.use('Agg')


def imshow(inp, title=None):
    """

    :param inp:
    :param title:
    :return:
    """

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_results(start=0, stop=0, bucket=''):
    train_iterations = [5, 4, 3, 2, 1]
    total_loss = [5, 4, 3, 2, 1]
    cout = 0

    # f, host = plt.subplots(1, 1)
    host = host_subplot(111)
    plt.subplots_adjust(right=0.8)

    print(cout)
    # set labels
    host.set_xlabel("Iterations")
    host.set_ylabel("total loss")

    # plot curves
    p1, = host.plot(train_iterations, total_loss, label="total_loss")
    host.legend(loc=5)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())

    host.set_xlim([-200, 5100])

    plt.draw()
    plt.show()

    # plt.tight_layout()
    # plt.savefig('results.png', dpi=200)
    host.savefig(os.path.join('reward' + '.png'), dpi=1000)


totalreward = [1, 1.2, 0.45, 0.23, 0.12]


def f():
    color = cm.viridis(0.5)
    f, ax = plt.subplots(1, 1)
    ax.plot(iter, totalreward, color=color)
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Return')
    exp_dir = 'Plot/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    else:
        os.makedirs(exp_dir, exist_ok=True)
    f.savefig(os.path.join('Plot', 'reward' + '.png'), dpi=1000)

# plot_results()
