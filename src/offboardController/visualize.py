# pip install visdom numpy
# visualize.py

import numpy
from visdom import Visdom

class Plot(object):
    def __init__(self, title, port=8097):
        self.viz = Visdom(port=port)
        self.windows = {}
        self.title = title

    def register_scatterplot(self, name, xlabel, ylabel):
        win = self.viz.scatter(
            X=numpy.zeros((1, 2)),
            opts=dict(title=self.title, markersize=4, xlabel=xlabel, ylabel=ylabel)
        )
        self.windows[name] = win

    def update_scatterplot(self, name, xlabel, ylabel, X, Y):
        self.viz.scatter(
            X,
            Y,
            win=self.windows[name],
            opts=dict(title=self.title, markersize=4, xlabel=xlabel, ylabel=ylabel, ytickmin=0,ytickmax=2,mode="markers+lines")
            #update='append'
        )

