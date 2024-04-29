import numpy as np
import matplotlib.pyplot as plt


filename = ""


def visclassifier(xTr, yTr, pred):
    # Plot the training data
    for i in range(xTr.shape[0]):
        symbol = "o" if yTr[i] == -1 else "x"
        if yTr[i] == pred[i]:
            plt.plot(xTr[i][0], xTr[i][1], symbol + "g")
        else:
            plt.plot(xTr[i][0], xTr[i][1], symbol + "r")

    # Set title and labels
    plt.title("Decision Tree Classifier")
    # Show the plot
    plt.show()


if __name__ == "__main__":
    pass
