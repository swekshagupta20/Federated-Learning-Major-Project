import matplotlib.pyplot as plt


def plot_accuracy(rounds, acc):
    plt.plot(rounds, acc)
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title("FL Convergence")
    plt.show()