"""
An alpha version of the Talbot, Lin, Hanrahan tick mark generator for matplotlib.
Described in "An Extension of Wilkinson's Algorithm for Positioning Tick Labels on Axes"
by Justin Talbot, Sharon Lin, and Pat Hanrahan, InfoVis 2010.

Implementation by Justin Talbot
This implementation is in the public domain.
Report bugs to jtalbot@stanford.edu

A shortcoming:
    The weights used in the paper were designed for static plots where the extent of
    the tick marks unioned with the extent of the data defines the extent of the plot.
    In a plot where the extent of the plot is defined by the user (e.g. an interactive
    plot supporting panning and zooming), the weights don't work as well. In particular,
    you would want to retune them assuming that the tick labels must be inside
    the provided view range. You probably want higher weighting on simplicity and lower
    on coverage and possibly density. But I haven't experimented in any detail with this.

    If you do intend on using this for static plots in matplotlib, you should set
    only_inside to False in the call to Extended.extended. And then you should
    manually set your view extent to include the min and max ticks if they are outside
    the data range. This should produce the same results as the paper.
"""

import math
import numpy as np

class Extended():
	# density is labels per inch
    def __init__(
            self, density = 1, steps = None, figure = None, 
            range=(0,1), axis = 'x'):
        """
        Keyword args:
        """
        self._density = density
        self._figure = figure
        self._axis = axis
        self.range=range

        if steps is None:
            self._steps = [1, 5, 2, 2.5, 4, 3]
        else:
            self._steps = steps


    def coverage(self, dmin, dmax, lmin, lmax):
        range = dmax-dmin
        return 1 - 0.5 * (math.pow(dmax-lmax, 2)+math.pow(dmin-lmin, 2)) / math.pow(0.1 * range, 2)

    def coverage_max(self, dmin, dmax, span):
        range = dmax-dmin
        if span > range:
            half = (span-range)/2.0
            return 1 - math.pow(half, 2) / math.pow(0.1*range, 2)
        else:
            return 1

    def density(self, k, m, dmin, dmax, lmin, lmax):
        r = (k-1.0) / (lmax-lmin)
        rt = (m-1.0) / (max(lmax, dmax) - min(lmin, dmin))
        return 2 - max( r/rt, rt/r )

    def density_max(self, k, m):
        if k >= m:
            return 2 - (k-1.0)/(m-1.0)
        else:
            return 1

    def simplicity(self, q, Q, j, lmin, lmax, lstep):
        eps = 1e-10
        n = len(Q)
        i = Q.index(q)+1
        v = 1 if ((lmin % lstep < eps or (lstep - lmin % lstep) < eps) and lmin <= 0 and lmax >= 0) else 0
        return (n-i)/(n-1.0) + v - j

    def simplicity_max(self, q, Q, j):
        n = len(Q)
        i = Q.index(q)+1
        v = 1
        return (n-i)/(n-1.0) + v - j

    def legibility(self, lmin, lmax, lstep):
        return 1

    def legibility_max(self, lmin, lmax, lstep):
        return 1

    def extended (self, dmin, dmax, m, Q=[1,5,2,2.5,4,3], only_inside=False, w=[0.25,0.2,0.5,0.05]):
        n = len(Q)
        best_score = -2.0

        j = 1.0
        while j < float('infinity'):
            for q in Q:
                sm = self.simplicity_max(q, Q, j)

                if w[0] * sm + w[1] + w[2] + w[3] < best_score:
                    j = float('infinity')
                    break

                k = 2.0
                while k < float('infinity'):
                    dm = self.density_max(k, m)

                    if w[0] * sm + w[1] + w[2] * dm + w[3] < best_score:
                        break

                    delta = (dmax-dmin)/(k+1.0)/j/q
                    z = math.ceil(math.log(delta, 10))
        
                    while z < float('infinity'):
                        step = j*q*math.pow(10,z)
                        cm = self.coverage_max(dmin, dmax, step*(k-1.0))

                        if w[0] * sm + w[1] * cm + w[2] * dm + w[3] < best_score:
                            break

                        min_start = math.floor(dmax/step)*j - (k-1.0)*j
                        max_start = math.ceil(dmin/step)*j

                        if min_start > max_start:
                            z = z+1
                            break

                        for start in range(int(min_start), int(max_start)+1):
                            lmin = start * (step/j)
                            lmax = lmin + step*(k-1.0)
                            lstep = step

                            s = self.simplicity(q, Q, j, lmin, lmax, lstep)
                            c = self.coverage(dmin, dmax, lmin, lmax)
                            d = self.density(k, m, dmin, dmax, lmin, lmax)
                            l = self.legibility(lmin, lmax, lstep)

                            score = w[0] * s + w[1] * c + w[2] * d + w[3] * l

                            if score > best_score and (not only_inside or (lmin >= dmin and lmax <= dmax)):
                                best_score = score
                                best = (lmin, lmax, lstep, q, k)
                        z = z+1
                    k = k+1
            j = j+1
        return best

    def __call__(self):
        vmin, vmax = self.range # self.axis.get_view_interval()
        fsize = {'x': 5., 'y': 4.}
        size = fsize[self._axis] # self._figure.get_size_inches()[self._axis]
		# density * size gives target number of intervals,
		# density * size + 1 gives target number of tick marks,
		# the density function converts this back to a density in data units (not inches)
		# should probably make this cleaner.
        best = self.extended(vmin, vmax, self._density * size + 1.0, only_inside=True, w=[0.25, 0.2, 0.5, 0.05])
        locs = np.arange(best[4]) * best[2] + best[0]
        return locs


if __name__ == '__main__':
    pass
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(10*np.random.randn(100), 10*np.random.randn(100), 'o')
    # 
    # xmin, xmax = ax.xaxis.get_data_interval()
    # xrange = xmax-xmin
    # xmin, xmax = (xmin - xrange * 0.05, xmax + xrange * 0.05)
    # 
    # ymin, ymax = ax.yaxis.get_data_interval()
    # yrange = ymax-ymin
    # ymin, ymax = (ymin - yrange * 0.05, ymax + yrange * 0.05)
    # 
    # ax.xaxis.set_view_interval(xmin, xmax, ignore=True)
    # ax.yaxis.set_view_interval(ymin, ymax, ignore=True)
    # ax.xaxis.set_major_locator(Extended(density=0.5, figure=fig, which=0))
    # ax.yaxis.set_major_locator(Extended(density=0.5, figure=fig, which=1))
    # 
    # ax.set_title('Talbot, Lin, Hanrahan 2010')
    # 
    # plt.show()