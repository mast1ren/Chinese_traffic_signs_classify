import matplotlib.pyplot as plt


def plotScore(plot_x, plot_acc, plot_recall, plot_precision, plot_f1, plot_x_loss, plot_loss):
    plt.figure(figsize=(10, 10), dpi=100)
    grid = plt.GridSpec(3, 2, wspace=0.2, hspace=0.5)
    plt.subplot(grid[0, 0])
    plt.plot(plot_x, plot_acc, 'o-b')
    plt.title('accurary', fontsize=20)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('accurary', fontsize=14)

    plt.subplot(grid[0, 1])
    plt.plot(plot_x, plot_recall, 'o-b')
    plt.title('recall', fontsize=20)
    plt.ylim(0, 1)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('recall', fontsize=14)

    plt.subplot(grid[1, 0])
    plt.plot(plot_x, plot_precision, 'o-b')
    plt.title('precision', fontsize=20)
    plt.ylim(0, 1)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('precision', fontsize=14)

    plt.subplot(grid[1, 1])
    plt.plot(plot_x, plot_f1, 'o-b')
    plt.ylim(0, 1)
    plt.title('f1', fontsize=20)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('f1', fontsize=14)

    plt.subplot(grid[2, 0:2])
    plt.plot(plot_x_loss, plot_loss, 'o-b')
    plt.title('loss', fontsize=20)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('loss', fontsize=14)

    plt.savefig(fname='score.svg', format='svg')

    plt.show()
