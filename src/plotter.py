from visdom import Visdom
import numpy as np

class EpochPlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', xlabel='Epochs'):
        self.viz = Visdom()
        self.env = env_name
        self.xlabel = xlabel
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]),
                Y=np.array([y,y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel=self.xlabel,
                    ylabel=var_name
                )
            )
        else:
            self.viz.line(X=np.array([x]),
                Y=np.array([y]), env=self.env,
                win=self.plots[var_name],
                name=split_name,
                update = 'append')
