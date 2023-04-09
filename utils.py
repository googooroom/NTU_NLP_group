
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plotter(data, fig_name):
    # calls 3 plot functions
    train_loss = data.get('train_loss') # using .get in case 
    train_acc = data.get('train_acc')
    val_loss = data.get('val_loss')
    val_acc = data.get('val_acc')

    # plot_loss_acc(train_loss, val_loss, train_acc, val_acc, fig_name)
    plot_acc(train_acc, val_acc, fig_name)
    plot_loss(train_loss, val_loss, fig_name)


# plot ONLY accuracy against epoch
def plot_acc(train_acc, val_acc, fig_name):
    new_fig_name = 'acc-' + fig_name
    x = np.arange(len(train_acc))
    
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')

    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    ax2.set_ylim([0,1])
    lns3 = ax2.plot(x, train_acc, 'bo-', label='train_acc')
    lns4 = ax2.plot(x, val_acc, 'ro-', label='val_acc')
    # ax2.tick_params(axis='y', labelcolor='tab:red')

    # lns = lns1+lns2+lns3+lns4
    lns = lns3+lns4
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=0)

    fig.tight_layout()
    plt.title(new_fig_name)

    plt.savefig(os.path.join('./diagram', new_fig_name))

def plot_loss(train_loss, val_loss, fig_name_ori):
    fig_name = 'loss-' + fig_name_ori
    x = np.arange(len(train_loss))
    max_loss = max(max(train_loss), max(val_loss))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_ylim([0,max_loss+1])
    lns1 = ax1.plot(x, train_loss, 'yo-', label='train_loss')
    lns2 = ax1.plot(x, val_loss, 'go-', label='val_loss')
    # ax1.tick_params(axis='y', labelcolor='tab:blue')

    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    fig.tight_layout()
    plt.title(fig_name)

    plt.savefig(os.path.join('./diagram', fig_name))


def epoch_time(start_time, end_time):
    """
    Convert epoch time to min and seconds
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
