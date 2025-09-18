from random import randint
from numpy import linspace, meshgrid, tan, exp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from numpy.random import random

pd.set_option('display.precision', 2)
pd.set_option('display.max_columns', None)
pd.options.display.expand_frame_repr = False
pd.options.mode.copy_on_write = True

def f(x1,x2):
    return (tan(x1)) / (3+exp(-2*x2))

def generate_csv(limit):
    x1 = linspace(-500, 500, limit)
    x2 = linspace(-1, 1, limit)
    y = f(x1,x2)
    dt = {'x1':x1, 'x2':x2, 'y':y}
    pd.DataFrame(dt).to_csv('lab2.csv', index=False)

def show_plt(data, const, y_label, x_label, flag):
    if not flag:
        y = f(data,const)
    else:
        y = f(const,data)

    plt.plot(data, y,marker='o', markersize=5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def graphics(data, limit):
    show_plt(data['x1'].head(limit), data['x2'][randint(0,len(data))], "y(x1)(x2-const)", "X1", False)
    show_plt(data['x2'].head(limit), data['x1'][randint(0,len(data))], "y(x2)(x1-const)", "X2",True)

def statistics(data):
    for i in ['x1','x2','y']:
        print(f"{i} min: {data[i].describe()['min']}\n"
              f"{i} max: {data[i].describe()['max']}\n"
              f"{i} mean: {data[i].describe()['mean']}\n")

def new_csv(data):
    mean_x1 = data['x1'].describe()['mean']
    mean_x2 = data['x2'].describe()['mean']
    dt = {'x1': [], 'x2': [], 'y': []}
    for i in range(len(data)):
        if data['x1'][i] < mean_x1 or data['x2'][i] < mean_x2:
            dt['x1'].append(data['x1'][i])
            dt['x2'].append(data['x2'][i])
            dt['y'].append(data['y'][i])
    pd.DataFrame(dt).to_csv('lab2_4.csv', index=False)

def graphisc3D(data):
    x1, x2 = meshgrid(data['x1'], data['x2'])
    y = f(x1, x2)
    ax = plt.axes(projection='3d')
    ax.plot_surface(x1, x2, y)
    plt.show()

def main():
    generate_csv(500)
    data = pd.read_csv('lab2.csv')

    graphics(data,500)
    statistics(data)
    new_csv(data)
    graphisc3D(data)
main()