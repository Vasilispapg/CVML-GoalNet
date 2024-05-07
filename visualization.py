from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def generate_metric_plots(opt_loss, est_train_losses, est_train_f_scores_avg, est_train_f_scores_max, val_losses, val_f_scores_avg, val_f_scores_max, exported_image_fp):

    baseline_loss = 0.3982

    n = len(est_train_losses)
    assert n == len(est_train_losses) == len(est_train_f_scores_avg) == len(est_train_f_scores_max) == len(val_losses) == len(val_f_scores_avg) == len(val_f_scores_max), 'E: Inconsistent score list lengths'

    x = list(range(n))

    fig = plt.figure(figsize = (10, 10), dpi = 190)
    fig.suptitle("Training states")

    ax = [[None], [None]]
    ax[0][0] = fig.add_subplot(2, 1, 1)
    ax[1][0] = fig.add_subplot(2, 1, 2)

    ax[0][0].plot(x, est_train_losses, color = 'orange', label = 'est_train')
    ax[0][0].plot(x, val_losses, color = 'cyan', label = 'val')
    ax[0][0].set_ylabel("Loss")
    # ax[0][0].set_xlabel("Epoch")
    ax[0][0].axhline(y = baseline_loss, color = 'purple', label = 'baseline %.4f'%(baseline_loss), linestyle = "dashed")
    ax[0][0].axhline(y = opt_loss, color = 'green', label = 'opt_val %.4f'%(opt_loss))
    # ax[0][0].set_ylim(0, 0.9)
    ax[0][0].legend()
    ax[0][0].grid()
    ax[0][0].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[1][0].plot(x, est_train_f_scores_avg, color = 'orange', label = 'est_train_avg')
    ax[1][0].plot(x, est_train_f_scores_max, color = 'red', label = 'est_train_max')
    ax[1][0].plot(x, val_f_scores_avg, color = 'cyan', label = 'val_avg')
    ax[1][0].plot(x, val_f_scores_max, color = 'blue', label = 'val_max')
    ax[1][0].set_ylabel("F-score")
    ax[1][0].set_xlabel("Epoch")
    ax[1][0].legend(fontsize=7)
    ax[1][0].grid()
    ax[1][0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # plt.show()
    plt.savefig(exported_image_fp)


if __name__ == "__main__":

    x = list(range(10))
    est_train_losses = est_train_f_scores_avg = est_train_f_scores_max = val_losses = val_f_scores_avg = val_f_scores_max = x
    generate_metric_plots(est_train_losses, est_train_f_scores_avg, est_train_f_scores_max, val_losses, val_f_scores_avg, val_f_scores_max)