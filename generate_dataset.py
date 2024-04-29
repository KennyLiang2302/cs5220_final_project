import numpy as np
import csv

N = 1000
filename_prefix = "dataset"


def spiraldata(N):
    # Credit to the CS 4780 GBRT Assignment
    r = np.linspace(1, 2 * np.pi, N)
    xTr1 = np.array([np.sin(2.0 * r) * r, np.cos(2 * r) * r]).T
    xTr2 = np.array([np.sin(2.0 * r + np.pi) * r, np.cos(2 * r + np.pi) * r]).T
    xTr = np.concatenate([xTr1, xTr2], axis=0)
    yTr = np.concatenate([np.ones(N), -1 * np.ones(N)])
    xTr = xTr + np.random.randn(xTr.shape[0], xTr.shape[1]) * 0.2

    print(xTr.shape)
    print(yTr.shape)
    return xTr, yTr


def write_to_csv(xTr, yTr):
    data = np.column_stack((xTr, yTr))
    data = data.tolist()

    filename = f"{filename_prefix}_{N}.csv"
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)


if __name__ == "__main__":
    xTr, yTr = spiraldata(N)
    write_to_csv(xTr, yTr)
