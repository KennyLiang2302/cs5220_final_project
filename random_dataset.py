import random
import csv

num_rows = 5000000
num_cols = 10
min_val = 0
max_val = 100

with open('datasets/random_5000000.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Label']
    writer.writerow(header)
    for _ in range(num_rows):
        row = [random.randint(min_val, max_val) for _ in range(num_cols)] + [random.randint(-1, 1)]
        writer.writerow(row)