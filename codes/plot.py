import matplotlib.pyplot as plt
import csv

plot_single = False
with open('output_cnn.csv') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    x_data = []
    cnn_y_data = []
    for row in reader:
        if i == 0:
            x_label, y_label = row[0], row[1]
        elif len(row) == 2:
            x_data.append(float(row[0]))
            cnn_y_data.append(float(row[1]))
        i += 1
if plot_single:
    plt.figure()
    plt.plot(x_data, cnn_y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('loss~iter')
    plt.savefig('cnn_plot.png')
else:
    with open('output_mlp.csv') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        x_data = []
        y_data = []
        for row in reader:
            if i == 0:
                x_label, y_label = row[0], row[1]
            elif len(row) == 2:
                x_data.append(float(row[0]))
                y_data.append(float(row[1]))
            i += 1
    plt.figure()
    plt.plot(x_data, y_data)
    plt.plot(x_data, cnn_y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('loss~iter')
    plt.savefig('compare.png')
