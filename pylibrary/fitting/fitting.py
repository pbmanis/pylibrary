"""
Python class wrapper for data fitting.
Includes the following external methods:
getFunctions returns the list of function names (dictionary keys)
FitRegion performs the fitting
Note that FitRegion will plot on top of the current data using MPlots routines
if the current curve and the current plot instance are passed.
January, 2009
Paul B. Manis, Ph.D.
UNC Chapel Hill
Department of Otolaryngology/Head and Neck Surgery
Supported by NIH Grants DC000425-22 and DC004551-07 to PBM.

MIT License (See LICENSE.txt for details)

"""

import numpy
import scipy
import scipy.optimize


# openoptFound = False
# try:
#     import openopt
#     openoptFound = True
# except Exception:
#     pass

import numpy.random

# from numba import autojit
usingMPlot = False

pyqt = False


class Fitting:
    # dictionary contains:
    # name of function: function call, initial parameters, iterations, plot color, then x and y for testing
    # target valutes, names of parameters, contant values, and derivative function if needed.
    #

    def __init__(self):
        self.fitfuncmap = {
            "exp0": (
                self.exp0eval,
                [0.0, 20.0],
                2000,
                "k",
                [0, 100, 1.0],
                [1.0, 5.0],
                ["A0", "tau"],
                None,
                None,
            ),
            "exp1": (
                self.expeval,
                [0.0, 0.0, 20.0],
                2000,
                "k",
                [0, 100, 1.0],
                [0.5, 1.0, 5.0],
                ["DC", "A0", "tau"],
                None,
                self.expevalprime,
            ),
            "expsat": (
                self.expsat,
                [0.0, 1.0, 20.0],
                2000,
                "k",
                [0, 10, 1.0],
                [0.5, 1.0, 5.0],
                ["DC", "A0", "tau"],
                None,
                self.expsatprime,
            ),
            "expsum": (
                self.expsumeval,
                [0.0, -0.5, 200.0, -0.25, 450.0],
                500000,
                "k",
                [0, 1000, 1.0],
                [0.0, -1.0, 150.0, -0.25, 350.0],
                ["DC", "A0", "tau0", "A1", "tau1"],
                None,
                None,
            ),
            "expsum2": (
                self.expsumeval2,
                [0.0, -0.5, -0.250],
                50000,
                "k",
                [0, 1000, 1.0],
                [0.0, -0.5, -0.25],
                ["A0", "A1"],
                [5.0, 20.0],
                None,
            ),
            "exp2": (
                self.exp2eval,
                [0.0, -0.5, 200.0, -0.25, 450.0],
                500000,
                "k",
                [0, 1000, 1.0],
                [0.0, -1.0, 150.0, -0.25, 350.0],
                ["DC", "A0", "tau0", "A1", "tau1"],
                None,
                None,
            ),
            "exppow": (
                self.exppoweval,
                [
                    0.0,
                    1.0,
                    100,
                ],
                2000,
                "k",
                [0, 100, 0.1],
                [0.0, 1.0, 100.0],
                ["DC", "A0", "tau"],
                None,
                None,
            ),
            "exppulse": (
                self.expPulse,
                [3.0, 2.5, 0.2, 2.5, 2.0, 0.5],
                2000,
                "k",
                [0, 10, 0.3],
                [0.0, 0.0, 0.75, 4.0, 1.5, 1.0],
                ["DC", "t0", "tau1", "tau2", "amp", "width"],
                None,
                None,
            ),
            "FIGrowth1": (
                self.FIGrowth1,
                [0.0, 100.0, 1.0, 40.0, 200.0],
                2000,
                "k",
                [0, 1000, 50],  # [Fzero, Ibreak, F1amp, F2amp, Irate]
                [0.0, 0.0, 0.0, 10.0, 100.0],
                ["Fzero", "Ibreak", "F1amp", "F2amp", "Irate"],
                None,
                None,
            ),
            "boltz": (
                self.boltzeval,
                [0.0, 1.0, -50.0, -5.0],
                5000,
                "r",
                [-130.0, -30.0, 1.0],
                [0.00, 0.010, -100.0, 7.0],
                ["DC", "A0", "x0", "k"],
                None,
                None,
            ),
            "normalizedgauss": (
                self.normalized_gausseval,
                [1.0, 0.0, 0.5],
                2000,
                "y",
                [-10.0, 10.0, 0.2],
                [1.0, 1.0, 2.0],
                ["A", "mu", "sigma"],
                None,
                None,
            ),
            "gauss": (
                self.gausseval,
                [1.0, 0.0, 0.5],
                2000,
                "y",
                [-10.0, 10.0, 0.2],
                [1.0, 1.0, 2.0],
                ["A", "mu", "sigma"],
                None,
                None,
            ),
            "ngauss": (
                self.ngausseval,
                [1.0, 0.0, 0.5, 2.0, 1.0, 0.25],
                2000,
                "y",
                [-10.0, 10.0, 0.2],
                [1.0, 1.0, 2.0, 0.5, 0.5, 0.1],
                ["A1", "mu1", "sigma1", "A2", "mu2", "sigma2"],
                2,
                None,
            ),
            "flattopgauss": (
                self.flattop_gausseval,
                [1.0, 0.0, 0.5, 0.5],
                50000,
                "y",
                [-10.0, 10.0, 0.01],
                [1.0, 1.0, 2.0, 0.2],
                ["A", "mu", "sigma", "ftwidth"],
                None,
                None,
            ),
            "flattopngauss": (
                self.flattop_ngausseval,
                [1.0, -1.0, 0.2, 0.25, 0.5, 1.0, 0.25, 0.5],
                50000,
                "y",
                [-10.0, 10.0, 0.01],
                [1.0, 1.0, 2.0, 0.2, 0.75, 1.0, 2.0, 0.2],
                ["A", "mu", "sigma", "ftwidth", "A2", "mu2", "sigma2", "ftwidth2"],
                2,
                None,
            ),
            "line": (
                self.lineeval,
                [1.0, 0.0],
                500,
                "r",
                [-10.0, 10.0, 0.5],
                [0.0, 2.0],
                ["m", "b"],
                None,
                None,
            ),
            "poly2": (
                self.poly2eval,
                [1.0, 1.0, 0.0],
                500,
                "r",
                [0, 100, 1.0],
                [0.5, 1.0, 5.0],
                ["a", "b", "c"],
                None,
                None,
            ),
            "poly3": (
                self.poly3eval,
                [1.0, 1.0, 1.0, 0.0],
                1000,
                "r",
                [0.0, 100.0, 1.0],
                [0.5, 1.0, 5.0, 2.0],
                ["a", "b", "c", "d"],
                None,
                None,
            ),
            "poly4": (
                self.poly4eval,
                [1.0, 1.0, 1.0, 1.0, 0.0],
                1000,
                "r",
                [0.0, 100.0, 1.0],
                [0.1, 0.5, 1.0, 5.0, 2.0],
                ["a", "b", "c", "d", "e"],
                None,
                None,
            ),
            "sin": (
                self.sineeval,
                [-1.0, 1.0, 4.0, 0.0],
                1000,
                "r",
                [0.0, 100.0, 0.2],
                [0.0, 1.0, 9.0, 0.0],
                ["DC", "A", "f", "phi"],
                None,
                None,
            ),
            "boltz2": (
                self.boltzeval2,
                [0.0, 0.5, -50.0, 5.0, 0.5, -20.0, 3.0],
                1200,
                "r",
                [-100.0, 50.0, 1.0],
                [0.0, 0.3, -45.0, 4.0, 0.7, 10.0, 12.0],
                ["DC", "A1", "x1", "k1", "A2", "x2", "k2"],
                None,
                None,
            ),
            "taucurve": (
                self.taucurve,
                [50.0, 300.0, 60.0, 10.0, 8.0, 65.0, 10.0],
                50000,
                "r",
                [-150.0, 50.0, 1.0],
                [0.0, 237.0, 60.0, 12.0, 17.0, 60.0, 14.0],
                ["DC", "a1", "v1", "k1", "a2", "v2", "k2"],
                None,
                self.taucurveder,
            ),
        }
        self.fitSum2Err = 0

    def getFunctions(self):
        return list(self.fitfuncmap.keys())

    def exp0eval(self, p, x, y=None, C=None, sumsq=False):
        """
        Exponential function with an amplitude and 0 offset
        """
        yd = p[0] * numpy.exp(-x / p[1])
        if y is None:
            return yd
        else:
            if sumsq is True:
                return numpy.sum((y - yd) ** 2)
            else:
                return y - yd

    def expsumeval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Sum of two exponentials with independent time constants and amplitudes,
        and a DC offset
        """
        yd = p[0] + (p[1] * numpy.exp(-x / p[2])) + (p[3] * numpy.exp(-x / p[4]))
        if y is None:
            return yd
        else:
            yerr = y - yd
            if weights is not None:
                yerr = yerr * weights
            if sumsq is True:
                return numpy.sum(yerr**2)
            else:
                return yerr

    def expsumeval2(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Sum of two exponential terms, with predefined time constants, allowing
        only the amplitudes and DC offset to vary
        """
        yd = p[0] + (p[1] * numpy.exp(-x / C[0])) + (p[2] * numpy.exp(-x / C[1]))
        if y is None:
            return yd
        else:
            if sumsq is True:
                return numpy.sum((y - yd) ** 2)
            else:
                return y - yd

    def expeval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Single exponential with DC offset
        """
        yd = p[0] + p[1] * numpy.exp(-x / p[2])
        if y is None:
            return yd
        else:
            if sumsq is True:
                return numpy.sum((y - yd) ** 2)
            else:
                return y - yd

    def expevalprime(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Derivative for exponential with offset
        """
        ydp = p[1] * numpy.exp(-x / p[2]) / (p[2] * p[2])
        yd = p[0] + p[1] * numpy.exp(-x / p[2])
        if y is None:
            return (yd, ydp)
        else:
            if sumsq is True:
                return numpy.sum((y - yd) ** 2)
            else:
                return y - yd

    def expsat(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Saturing single exponential rise with DC offset:
        p[0] + p[1]*(1-np.exp(-x/p[2]))
        """
        yd = p[0] + p[1] * (1.0 - numpy.exp(-x * p[2]))
        if y is None:
            return yd
        else:
            if sumsq is True:
                return numpy.sum((y - yd) ** 2)
            else:
                return y - yd

    def expsatprime(self, p, x):
        """
        derivative for expsat
        """
        #        yd = p[0] + p[1] * (1.0 - numpy.exp(-x * p[2]))
        ydp = p[1] * p[2] * numpy.exp(-x * p[2])
        return ydp

    def exppoweval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Single exponential function, raised to a power
        """
        if C is None:
            cx = 1.0
        else:
            cx = C[0]
        yd = p[0] + p[1] * (1.0 - numpy.exp(-x / p[2])) ** cx
        if y is None:
            return yd
        else:
            if sumsq is True:
                return numpy.sum((y - yd) ** 2)
            else:
                return y - yd

    def exp2eval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        For fit to activation currents, 2nd order exp + first order exp
        """
        yd = (
            p[0]
            + (p[1] * (1.0 - numpy.exp(-x / p[2])) ** 2.0)
            + (p[3] * (1.0 - numpy.exp(-x / p[4])))
        )
        if y is None:
            return yd
        else:
            if sumsq is True:
                ss = numpy.sqrt(numpy.sum((y - yd) ** 2.0))
                #                if p[4] < 3.0*p[2]:
                # ss = ss*1e6 # penalize them being too close
                return ss
            else:
                return y - yd

            #    @autojit

    def expPulse(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Exponential pulse function (rising exponential with variable-length
        plateau followed by falling exponential)
        Parameter p is [offset, tau1, tau2, amp, width]
        """
        yOffset, t0, tau1, tau2, amp, width = p
        yd = numpy.empty(x.shape)
        yd[x < t0] = yOffset
        m1 = (x >= t0) & (x < (t0 + width))
        m2 = x >= (t0 + width)
        x1 = x[m1]
        x2 = x[m2]
        yd[m1] = amp * (1 - numpy.exp(-(x1 - t0) / tau1)) + yOffset
        # y-value at start of decay
        amp2 = amp * (1 - numpy.exp(-width / tau1))
        yd[m2] = ((amp2) * numpy.exp(-(x2 - (width + t0)) / tau2)) + yOffset
        if y is None:
            return yd
        else:
            if sumsq is True:
                ss = numpy.sqrt(numpy.sum((y - yd) ** 2.0))
                return ss
            else:
                return y - yd

    def FIGrowth1(self, p, x, y=None, C=None, sumsq=True, weights=None):
        """
        Frequency versus current intensity (FI plot) fit
        Linear fit from 0 to breakpoint
        exponential growth thereafter

        Parameter p is a list containing: [Fzero, Ibreak, F1amp, F2amp, Irate]
        for I < break: F = Fzero + I*F1amp
        for I >= break: F = F(break)+ F2amp(1-exp^(-t * Irate))
        """
        Fzero, Ibreak, F1amp, F2amp, Irate = p
        yd = numpy.zeros(x.shape)
        m1 = x < Ibreak
        m2 = x >= Ibreak
        yd[m1] = Fzero + x[m1] * F1amp / Ibreak
        maxyd = numpy.max(yd)
        yd[m2] = F2amp * (1.0 - numpy.exp(-(x[m2] - Ibreak) * Irate)) + maxyd
        if y is None:
            return yd
        else:
            dy = y - yd
            w = numpy.ones(len(x))
            #            xp = numpy.argwhere(x>0)
            #            w[xp] = w[xp] + 3.*x[xp]/numpy.max(x)
            if sumsq is True:
                ss = numpy.sqrt(numpy.sum((w * dy) ** 2.0))
                return ss
            else:
                return w * dy

    def boltzeval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        yd = p[0] + (p[1] - p[0]) / (1.0 + numpy.exp((x - p[2]) / p[3]))
        if y is None:
            return yd
        else:
            if sumsq is True:
                return numpy.sqrt(numpy.sum((y - yd) ** 2.0))
            else:
                return y - yd

    def boltzeval2(self, p, x, y=None, C=None, sumsq=False, weights=None):
        yd = (
            p[0]
            + p[1] / (1 + numpy.exp((x - p[2]) / p[3]))
            + p[4] / (1 + numpy.exp((x - p[5]) / p[6]))
        )
        if y is None:
            return yd
        else:
            if sumsq is True:
                return numpy.sum((y - yd) ** 2)
            else:
                return y - yd

    def normalized_gausseval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Normalized version (area = 1)
        """
        yd = (p[0] / (p[2] * numpy.sqrt(2.0 * numpy.pi))) * numpy.exp(
            -((x - p[1]) ** 2.0) / (2.0 * (p[2] ** 2.0))
        )
        if y is None:
            return yd
        else:
            if sumsq is True:
                return numpy.sum((y - yd) ** 2)
            else:
                return y - yd

    def gausseval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Non-normalized version
        """
        yd = p[0] * numpy.exp(-((x - p[1]) ** 2.0) / (2.0 * (p[2] ** 2.0)))
        if y is None:
            return yd
        else:
            if sumsq is True:
                return numpy.sum((y - yd) ** 2)
            else:
                return y - yd

    def flattop_gausseval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        """
        Non-normalized version
        p[0] is amplitude (a)
        p[1] is central position (x0)
        p[2] is sigma (signma)
        p[3] is the half-width of the flattop region (w) (should be >= 0.)
        """
        if p[3] <= 0.0:
            return self.gausseval(p[0:3], x, y=y, C=C, sumsq=sumsq, weights=weights)

        pl = p[1] - p[3]
        pr = p[1] + p[3]
        u = 1.0 / (2.0 * (p[2] * p[2]))
        #        eu = numpy.exp(u)
        xl = numpy.argwhere(x <= pl).flatten()  # all to the left of the ft
        # pts to the right of the flattop
        xr = numpy.argwhere(x >= pr).flatten()
        #        A = p[0] # / (numpy.sqrt(p[2] * 2.0 * numpy.pi))  # no need to scale...
        yd = p[0] * numpy.ones(x.shape)
        al = x[xl] - pl
        ar = x[xr] - pr
        # yd[xl] = A * norm.pdf(al, 0., p[2])
        yd[xl] = p[0] * numpy.exp(-numpy.square(al) * u)
        # yd[xr] = A * norm.pdf(ar, 0., p[2])
        yd[xr] = p[0] * numpy.exp(-numpy.square(ar) * u)
        # flatpts = numpy.argwhere((x > pl) & (x < pr)).flatten() # if inside the flatop interval
        #        if len(flatpts) > 0:
        #            yd[flatpts[0]:flatpts[-1]+1] = p[0]*numpy.ones(len(flatpts))
        #            yd[flatpts] = p[0]*numpy.ones(len(flatpts))
        if y is None:
            return yd
        else:
            if sumsq is True:
                return numpy.sum((y - yd) ** 2)
            else:
                return y - yd

    def ngausseval(self, p, x, y=None, C=2, sumsq=False, weights=None):
        # C is the number of gaussians to evaluate
        # p[0,1,2] are the parameters for the first gaussian
        # p[3,4,5] are the parameters for the second
        # etc.
        #
        for i in range(int(C)):
            pn = p[(i * 3) : (i * 3) + 3]
            if i == 0:
                yd = self.gausseval(pn, x, y=None)
            else:
                yd += self.gausseval(pn, x, y=None)
        if y is None:
            return yd
        else:
            if sumsq is True:
                return numpy.sum((y - yd) ** 2)
            else:
                return y - yd

    def flattop_ngausseval(self, p, x, y=None, C=2, sumsq=False, weights=None):
        # C[0] is the number of gaussians to evaluate
        # C[1] is the truncation value (usually, 1.0)
        # p[0,1,2,3] are the parameters for the first gaussian
        # p[4,5,6,7] are the parameters for the second
        # etc.
        #
        for i in range(int(C)):
            pn = p[(i * 4) : (i * 4) + 4]
            if i == 0:
                yd = self.flattop_gausseval(pn, x, y=None, sumsq=sumsq)
            else:
                yd += self.flattop_gausseval(pn, x, y=None, sumsq=sumsq)
        if y is None:
            return yd
        else:
            if sumsq is True:
                return numpy.sum((y - yd) ** 2.0)
            else:
                return y - yd

    def lineeval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        yd = p[0] * x + p[1]
        if y is None:
            return yd
        else:
            if sumsq is True:
                return numpy.sum((y - yd) ** 2)
            else:
                return y - yd

    def poly2eval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        yd = p[0] * x**2.0 + p[1] * x + p[2]
        if y is None:
            return yd
        else:
            if sumsq is True:
                return numpy.sum((y - yd) ** 2)
            else:
                return y - yd

    def poly3eval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        yd = p[0] * x**3.0 + p[1] * x**2.0 + p[2] * x + p[3]
        if y is None:
            return yd
        else:
            if sumsq is True:
                return numpy.sum((y - yd) ** 2)
            else:
                return y - yd

    def poly4eval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        yd = p[0] * x**4.0 + p[1] * x**3.0 + p[2] * x**2.0 + p[3] * x + p[4]
        if y is None:
            return yd
        else:
            if sumsq is True:
                return numpy.sum((y - yd) ** 2)
            else:
                return y - yd

    def sineeval(self, p, x, y=None, C=None, sumsq=False, weights=None):
        yd = p[0] + p[1] * numpy.sin((x * 2.0 * numpy.pi / p[2]) + p[3])
        if y is None:
            return yd
        else:
            if sumsq is True:
                return numpy.sum((y - yd) ** 2)
            else:
                return y - yd

    def taucurve(self, p, x, y=None, C=None, sumsq=True, weights=None):
        """
        HH-like description of activation/inactivation function
        'DC', 'a1', 'v1', 'k1', 'a2', 'v2', 'k2'
        """
        yd = p[0] + 1.0 / (
            p[1] * numpy.exp((x + p[2]) / p[3]) + p[4] * numpy.exp(-(x + p[5]) / p[6])
        )
        if y is None:
            return yd
        else:
            if sumsq is True:
                return numpy.sqrt(numpy.sum((y - yd) ** 2))
            else:
                return y - yd

    def taucurveder(self, p, x):
        """
        Derivative for taucurve
        'DC', 'a1', 'v1', 'k1', 'a2', 'v2', 'k2'
        """
        y = (
            -(
                p[1] * numpy.exp((p[2] + x) / p[3]) / p[3]
                - p[4] * numpy.exp(-(p[5] + x) / p[6]) / p[6]
            )
            / (p[1] * numpy.exp((p[2] + x) / p[3]) + p[4] * numpy.exp(-(p[5] + x) / p[6])) ** 2.0
        )
        #  print 'dy: ', y
        return y

    def getClipData(self, x, y, t0, t1):
        """
        Return the values in y that match the x range in tx from
        t0 to t1. x must be monotonic increasing or decreasing.
        Allow for reverse ordering.
        """
        it0 = (numpy.abs(x - t0)).argmin()
        it1 = (numpy.abs(x - t1)).argmin()
        if it0 > it1:
            t = it1
            it1 = it0
            it0 = t
        return (x[it0:it1], y[it0:it1])

    def FitRegion(
        self,
        whichdata,
        thisaxis,
        tdat,
        ydat,
        t0=None,
        t1=None,
        fitFunc="exp1",
        fitFuncDer=None,
        fitPars=None,
        fixedPars=None,
        fitPlot=None,
        plotInstance=None,
        dataType="xy",
        method=None,
        bounds=None,
        weights=None,
        constraints=(),
    ):
        """
        Fit x,y data pair over a region of x
        To call with tdat and ydat as simple arrays:
        FitRegion(1, 0, tdat, ydat, FitFunc = 'exp1')
        e.g., the first argument should be 1, but this axis is ignored if datatype is 'xy'
        """
        self.fitSum2Err = 0.0
        if t0 == t1:
            if plotInstance is not None and usingMPlot:
                (x, y) = plotInstance.getCoordinates()
                t0 = x[0]
                t1 = x[1]
        if t1 is None:
            t1 = numpy.max(tdat)
        if t0 is None:
            t0 = numpy.min(tdat)
        func = self.fitfuncmap[fitFunc]
        if func is None:
            print("FitRegion: unknown function %s" % (fitFunc))
            return
        xp = []
        xf = []
        yf = []
        yn = []
        tx = []
        names = func[6]
        if fitPars is None:
            fpars = func[1]
        else:
            fpars = fitPars
        # remap calls if needed for newer versions of scipy (>= 0.11)
        if method == "simplex":
            method = "Nelder-Mead"
        # check if 1-d, then "pretend" its only a 1-element block
        if ydat.ndim == 1 or dataType == "xy" or dataType == "2d":
            nblock = 1
        else:
            # otherwise, this is the number of traces in the block
            nblock = ydat.shape[0]
        for block in range(nblock):
            for record in whichdata:
                if dataType == "blocks":
                    (tx, dy) = self.getClipData(
                        tdat[block], ydat[block][record, thisaxis, :], t0, t1
                    )
                else:
                    if ydat.ndim > 1:
                        (tx, dy) = self.getClipData(tdat, ydat[record, :], t0, t1)
                    else:
                        (tx, dy) = self.getClipData(tdat, ydat, t0, t1)

                        #  print 'Fitting.py: Fit data: ', tx, dy
                yn.append(names)
                if not any(tx):
                    continue  # no data in the window...
                ier = 0
                #
                # Different optimization methods are included here. Not all have been tested fully with
                # this wrapper.
                #
                # use standard leastsq, no bounds
                if method is None or method == "leastsq":
                    plsq, cov, infodict, mesg, ier = scipy.optimize.leastsq(
                        func[0],
                        fpars,
                        args=(tx.astype("float64"), dy.astype("float64"), fixedPars),
                        full_output=1,
                        maxfev=func[2],
                    )
                    # print 'none/leastsq'
                    # print 'plsq: ', plsq
                    # print 'cov: ', cov
                    if ier > 4:
                        print("optimize.leastsq error flag is: %d" % (ier))
                        print(mesg)
                elif method == "curve_fit":
                    # print fpars
                    # print fixedPars
                    plsq, cov = scipy.optimize.curve_fit(
                        func[0], tx.astype("float64"), dy.astype("float64"), p0=fpars
                    )
                    # print 'curve_fit'
                    # print 'plsq: ', plsq
                    # print 'cov: ', cov
                    ier = 0
                elif method in [
                    "fmin",
                    "simplex",
                    "Nelder-Mead",
                    "bfgs",
                    "TNC",
                    "SLSQP",
                    "COBYLA",
                    "L-BFGS-B",
                ]:  # use standard wrapper from scipy for those routintes
                    res = scipy.optimize.minimize(
                        func[0],
                        fpars,
                        args=(tx.astype("float64"), dy.astype("float64"), fixedPars, True),
                        method=method,
                        jac=None,
                        hess=None,
                        hessp=None,
                        bounds=bounds,
                        constraints=constraints,
                        tol=None,
                        callback=None,
                        options={"maxiter": func[2], "disp": False},
                    )
                    plsq = res.x
                    # print 'res: ', res

                # next section is replaced by the code above - kept here for reference if needed...
                # elif method == 'fmin' or method == 'simplex':
                #     plsq = scipy.optimize.fmin(func[0], fpars, args=(tx.astype('float64'), dy.astype('float64'), fixedPars, True),
                # maxfun = func[2]) # , iprint=0)
                #     ier = 0
                # elif method == 'bfgs':
                #     plsq, cov, infodict = scipy.optimize.fmin_l_bfgs_b(func[0], fpars, fprime=func[8],
                #                 args=(tx.astype('float64'), dy.astype('float64'), fixedPars, True, weights),
                #                 maxfun = func[2], bounds = bounds,
                # approx_grad = True) # , disp=0, iprint=-1)
                # use OpenOpt's routines - usually slower, but sometimes they
                # converge better
                # elif method == 'openopt' and openoptFound:
                #     print('OpenOpt!!!')
                #     if bounds is not None:
                #         # unpack bounds
                #         lb = [z[0] for z in bounds]
                #         ub = [z[1] for z in bounds]
                #         fopt = openopt.DFP(
                #             func[0], fpars, tx, dy, df=fitFuncDer, lb=lb, ub=ub)
                #         # fopt.df = func[8]
                #         r = fopt.solve('nlp:ralg', plot=0, iprint=10)
                #         plsq = r.xf
                #         ier = 0
                #     else:
                #         fopt = openopt.DFP(
                #             func[0], fpars, tx, dy, df=fitFuncDer)
                #         print(func[8])
                #         #  fopt.df = func[7]
                #         fopt.checkdf()
                #         r = fopt.solve('nlp:ralg', plot=0, iprint=10)
                #         plsq = r.xf
                #         ier = 0
                else:
                    print("method %s not recognized, please check Fitting.py" % (method))
                    return
                # min(tx), max(tx), (max(tx)-min(tx))/100.0)
                xfit = numpy.linspace(t0, t1, num=100, endpoint=True)
                yfit = func[0](plsq, xfit, C=fixedPars)
                yy = func[0](plsq, tx, C=fixedPars)  # calculate function
                self.fitSum2Err = numpy.sum((dy - yy) ** 2)
                if usingMPlot and fitPlot != None and plotInstance != None:
                    self.FitPlot(
                        xFit=xfit,
                        yFit=yfit,
                        fitFunc=func[0],
                        fitPars=plsq,
                        plot=fitPlot,
                        plotInstance=plotInstance,
                    )
                xp.append(plsq)  # parameter list
                xf.append(xfit)  # x plot point list
                yf.append(yfit)  # y fit point list
        return (xp, xf, yf, yn)  # includes names with yn and range of tx

    def FitPlot(
        self,
        xFit=None,
        yFit=None,
        fitFunc="exp1",
        fitPars=None,
        fixedPars=None,
        fitPlot=None,
        plotInstance=None,
        color=None,
    ):
        """
        Plot the fit data onto the fitPlot with the specified "plot Instance".
             if there is no xFit, or some parameters are missing, we just return.
             if there is xFit, but no yFit, then we try to compute the fit with
             what we have. The plot is superimposed on the specified "fitPlot" and
             the color is specified by the function color in the fitPars list.
        """
        if xFit is None or fitPars is None:
            return
        func = self.fitfuncmap[fitFunc]
        if color is None:
            fcolor = func[3]
        else:
            fcolor = color
        if yFit is None:
            yFit = numpy.array(xFit.shape)
            #            print 'xfit shape: ', xFit.shape
            yFit = func[0](fitPars, xFit, C=fixedPars)
        #  for k in range(0, len(fitPars)):
        #     yFit[k] = func[0](fitPars[k], xFit[k], C=fixedPars)
        if plotInstance is None or fitPlot is None:
            return yFit
        for k in range(0, len(fitPars)):
            plotInstance.PlotLine(fitPlot, xFit[k], yFit[k], color=fcolor)
        return yFit

    def getFitErr(self):
        """Return the fit error for the most recent fit"""
        return self.fitSum2Err

    def expfit(self, x, y):
        """find best fit of a single exponential function to x and y
        using the chebyshev polynomial approximation.
        returns (DC, A, tau) for fit.

        Perform a single exponential fit to data using Chebyshev polynomial method.
        Equation fit: y = a1 * exp(-x/tau) + a0
        Call: [a0 a1 tau] = expfit(x,y);
        Calling parameter x is the time base, y is the data to be fit.
        Returned values: a0 is the offset, a1 is the amplitude, tau is the time
        constant (scaled in units of x).
        Relies on routines chebftd to generate polynomial coeffs, and chebint to compute the
        coefficients for the integral of the data. These are now included in this
        .py file source.
        This version is based on the one in the pClamp manual: HOWEVER, since
        I use the bounded [-1 1] form for the Chebyshev polynomials, the coefficients are different,
        and the resulting equation for tau is different. I manually optimized the tau
        estimate based on fits to some simulated noisy data. (Its ok to use the whole range of d1 and d0
        when the data is clean, but only the first few coeffs really hold the info when
        the data is noisy.)
        NOTE: The user is responsible for making sure that the passed data is appropriate,
        e.g., no large noise or electronic transients, and that the time constants in the
        data are adequately sampled.
        To do a double exp fit with this method is possible, but more complex.
        It would be computationally simpler to try breaking the data into two regions where
        the fast and slow components are dominant, and fit each separately; then use that to
        seed a non-linear fit (e.g., L-M) algorithm.
        Final working version 4/13/99 Paul B. Manis
        converted to Python 7/9/2009 Paul B. Manis. Seems functional.
        """
        n = 30  # default number of polynomials coeffs to use in fit
        a = numpy.amin(x)
        b = numpy.amax(x)
        d0 = self.chebftd(a, b, n, x, y)  # coeffs for data trace...
        d1 = self.chebint(a, b, d0, n)  # coeffs of integral...
        tau = -numpy.mean(d1[2:3] / d0[2:3])
        try:
            g = numpy.exp(-x / tau)
        except:
            g = 0.0
        # generate chebyshev polynomial for unit exponential function
        dg = self.chebftd(a, b, n, x, g)
        # now estimate the amplitude from the ratios of the coeffs.
        a1 = self.estimate(d0, dg, 1)
        a0 = (d0[0] - a1 * dg[0]) / 2.0  # get the offset here
        return (a0, a1, tau)

    def estimate(self, c, d, m):
        """compute optimal estimate of parameter from arrays of data"""
        n = len(c)
        a = sum(c[m:n] * d[m:n]) / sum(d[m:n] ** 2.0)
        return a

    # note : the following routine is a bottleneck. It should be coded in C.

    def chebftd(self, a, b, n, t, d):
        """Chebyshev fit; from Press et al, p 192.
        matlab code P. Manis 21 Mar 1999
        "Given a function func, lower and upper limits of the interval [a,b], and
        a maximum degree, n, this routine computes the n coefficients c[1..n] such that
        func(x) sum(k=1, n) of ck*Tk(y) - c0/2, where y = (x -0.5*(b+a))/(0.5*(b-a))
        This routine is to be used with moderately large n (30-50) the array of c's is
        subsequently truncated at the smaller value m such that cm and subsequent
        terms are negligible."
        This routine is modified so that we find close points in x (data array) - i.e., we find
        the best Chebyshev terms to describe the data as if it is an arbitrary function.
        t is the x data, d is the y data...
        """
        bma = 0.5 * (b - a)
        bpa = 0.5 * (b + a)
        inc = t[1] - t[0]
        f = numpy.zeros(n)
        for k in range(0, n):
            y = numpy.cos(numpy.pi * (k + 0.5) / n)
            pos = int(0.5 + (y * bma + bpa) / inc)
            if pos < 0:
                pos = 0
            if pos >= len(d) - 2:
                pos = len(d) - 2
            try:
                f[k] = d[pos + 1]
            except:
                print(
                    "error in chebftd: k = %d (len f = %d)  pos = %d, len(d) = %d\n"
                    % (k, len(f), pos, len(d))
                )
                print("you should probably make sure this doesn't happen")
        fac = 2.0 / n
        c = numpy.zeros(n)
        for j in range(0, n):
            sum = 0.0
            for k in range(0, n):
                sum = sum + f[k] * numpy.cos(numpy.pi * j * (k + 0.5) / n)
            c[j] = fac * sum
        return c

    def chebint(self, a, b, c, n):
        """Given a, b, and c[1..n] as output from chebft or chebftd, and given n,
        the desired degree of approximation (length of c to be used),
        this routine computes cint, the Chebyshev coefficients of the
        integral of the function whose coeffs are in c. The constant of
        integration is set so that the integral vanishes at a.
        Coded from Press et al, 3/21/99 P. Manis (Matlab)
        Python translation 7/8/2009 P. Manis
        """
        sum = 0.0
        fac = 1.0
        con = 0.25 * (b - a)  # factor that normalizes the interval
        cint = numpy.zeros(n)
        for j in range(1, n - 2):
            cint[j] = con * (c[j - 1] - c[j + 1]) / j
            sum = sum + fac * cint[j]
            fac = -fac
        cint[n - 1] = con * c[n - 2] / (n - 1)
        sum = sum + fac * cint[n - 1]
        cint[0] = 2.0 * sum  # set constant of integration.
        return cint

    def flatten(self, l, ltypes=(list, tuple)):
        """
        routine to flatten an array/list.
        """
        i = 0
        while i < len(l):
            while isinstance(l[i], ltypes):
                if not l[i]:
                    l.pop(i)
                    if not len(l):
                        break
                else:
                    l[i : i + 1] = list(l[i])
            i += 1
        return l


# run some tests if we are "main"


def main():
    #    import matplotlib.pyplot as pyplot

    import matplotlib.pyplot as pylab

    #
    # pylab.rcParams['text.usetex'] = True
    # pylab.rcParams['interactive'] = False
    # pylab.rcParams['font.family'] = 'sans-serif'
    # pylab.rcParams['font.sans-serif'] = 'Arial'
    # pylab.rcParams['mathtext.default'] = 'sf'
    # pylab.rcParams['figure.facecolor'] = 'white'
    # next setting allows pdf font to be readable in Adobe Illustrator
    # pylab.rcParams['pdf.fonttype'] = 42
    # pylab.rcParams['text.dvipnghack'] = True
    # to here (matplotlib stuff - touchy!

    print("Testing Fitting")
    Fits = Fitting()
    #    x = numpy.arange(0, 100.0, 0.1)
    #    y = 5.0-2.5*numpy.exp(-x/5.0)+0.5*numpy.random.randn(len(x))
    #    (dc, aFit,tauFit) = Fits.expfit(x,y)
    #    yf = dc + aFit*numpy.exp(-x/tauFit)
    #   pyplot.figure(1)
    #  pyplot.plot(x,y,'k')
    #  pyplot.hold(True)
    #  pyplot.plot(x, yf, 'r')
    #  pyplot.show()
    exploreError = False

    func = "flattopngauss"
    if exploreError is True:
        # explore the error surface for a function:

        f = Fits.fitfuncmap[func]
        p1range = numpy.arange(0.1, 5.0, 0.1)
        p2range = numpy.arange(0.1, 5.0, 0.1)

        err = numpy.zeros((len(p1range), len(p2range)))
        x = numpy.array(numpy.arange(f[4][0], f[4][1], f[4][2]))
        C = None
        if func in ["expsum2", "exppulse", "ngauss"]:
            C = f[7]
        # check exchange of tau1 ([1]) and width[4]
        yOffset, t0, tau1, tau2, amp, width = f[1]  # get inital parameters
        y0 = f[0](f[1], x, C=C)
        noise = numpy.random.random(y0.shape) - 0.5
        y0 += 0.0 * noise
        sh = err.shape
        yp = numpy.zeros((sh[0], sh[1], len(y0)))
        for i, p1 in enumerate(p1range):
            tau1t = tau1 * p1
            for j, p2 in enumerate(p2range):
                ampt = amp * p2
                pars = (yOffset, t0, tau1t, tau2, ampt, width)  # repackage
                err[i, j] = f[0](pars, x, y0, C=C, sumsq=True)
                yp[i, j] = f[0](pars, x, C=C, sumsq=False)

        pylab.figure()
        CS = pylab.contour(p1range * tau1, p2range * width, err, 25)
        CB = pylab.colorbar(CS, shrink=0.8, extend="both")
        pylab.figure()
        for i, p1 in enumerate(p1range):
            for j, p2 in enumerate(p2range):
                pylab.plot(x, yp[i, j])
        pylab.plot(x, y0, "r-", linewidth=2.0)

    # run tests for each type of fit, return results to compare parameters

    cons = []
    bnds = None

    signal_to_noise = 100000.0
    for func in Fits.fitfuncmap:
        if func != "flattopngauss":
            continue
        print("\nFunction: %s\nTarget: " % (func), end=" ")
        f = Fits.fitfuncmap[func]
        for k in range(0, len(f[1])):
            print("%f " % (f[1][k]), end=" ")
        print("\nStarting:     ", end=" ")
        for k in range(0, len(f[5])):
            print("%f " % (f[5][k]), end=" ")

        #        nstep = 500.0
        #        if func == 'sin':
        #            nstep = 100.0
        x = numpy.array(numpy.arange(f[4][0], f[4][1], f[4][2]))
        C = None
        if func in ["expsum2", "exppulse", "ngauss", "flattopgauss", "flattopngauss"]:
            print("\n", f[7])
            C = f[7]

        y = f[0](f[1], x, C=C)
        yd = numpy.array(y)
        noise = numpy.random.normal(0, 0.1, yd.shape)
        my = numpy.amax(yd)
        # yd = yd + sigmax*0.05*my*(numpy.random.random_sample(shape(yd))-0.5)
        yd += noise * my / signal_to_noise
        testMethod = "SLSQP"
        if func == "taucurve":
            continue
            bounds = [
                (0.0, 100.0),
                (0.0, 1000.0),
                (0.0, 500.0),
                (0.1, 50.0),
                (0.0, 1000),
                (0.0, 500.0),
                (0.1, 50.0),
            ]
            (fpar, xf, yf, names) = Fits.FitRegion(
                numpy.array([1]), 0, x, yd, fitFunc=func, bounds=bounds, method=testMethod
            )
        elif func == "FIGrowth1":
            # Parameter p is [Fzero, Ibreak, F1amp, F2amp, Irate]
            print("\nFIGrowth1: ")
            bounds = [
                (0.0, 3.0),
                (0.0, 1000),
                (0.0, 5.0),
                (
                    0.0,
                    100.0,
                ),
                (0.0, 100.0),
            ]
            initialgr = f[0](f[5], x, None)
            (fpar, xf, yf, names) = Fits.FitRegion(
                numpy.array([1]),
                0,
                x,
                yd,
                fitFunc=func,
                bounds=bounds,
                constraints=[],
                fixedPars=None,
                method=testMethod,
            )
            tv = f[5]

        elif func == "boltz":
            bounds = [(-0.5, 0.5), (0.0, 20.0), (-120.0, 0.0), (-20.0, 0.0)]
            (fpar, xf, yf, names) = Fits.FitRegion(
                numpy.array([1]), 0, x, yd, fitFunc=func, bounds=bounds, method=testMethod
            )

        elif func == "exp2":
            bounds = [(-0.001, 0.001), (-5.0, 0.0), (1.0, 500.0), (-5.0, 0.0), (1.0, 10000.0)]
            (fpar, xf, yf, names) = Fits.FitRegion(
                numpy.array([1]), 0, x, yd, fitFunc=func, bounds=bounds, method=testMethod
            )

        elif func == "exppulse":
            # set some constraints to the fitting
            # yOffset, tau1, tau2, amp, width = f[1]  # order of constraings
            dt = numpy.mean(numpy.diff(x))
            bounds = [(-5, 5), (-15.0, 15.0), (-2, 2.0), (2 - 10, 10.0), (-5, 5.0), (0.0, 5.0)]
            # cxample for constraints:
            # cons = ({'type': 'ineq', 'fun': lambda x:   x[4] - 3.0*x[2]},
            #         {'type': 'ineq', 'fun': lambda x:   - x[4] + 12*x[2]},
            #         {'type': 'ineq', 'fun': lambda x:   x[2]},
            #         {'type': 'ineq', 'fun': lambda x:  - x[4] + 2000},
            #         )
            cons = ({"type": "ineq", "fun": lambda x: x[3] - x[2]},)  # tau1 < tau2
            C = None

            tv = f[5]
            initialgr = f[0](f[5], x, None)
            (fpar, xf, yf, names) = Fits.FitRegion(
                numpy.array([1]),
                0,
                x,
                yd,
                fitFunc=func,
                fixedPars=C,
                constraints=cons,
                bounds=bounds,
                method=testMethod,
            )
            # print xf
            # print yf
            # print fpar
            # print names

        else:
            tv = f[5]
            initialgr = f[0](f[5], x, None)
            C = [1, 1]
            (fpar, xf, yf, names) = Fits.FitRegion(
                numpy.array([1]),
                0,
                x,
                yd,
                fitFunc=func,
                fixedPars=C,
                constraints=cons,
                bounds=bnds,
                method=testMethod,
            )
            # print fpar
        s = numpy.shape(fpar)
        j = 0
        outstr = ""
        initstr = ""
        truestr = ""
        for i in range(0, len(names[j])):
            #            print "%f " % fpar[j][i],
            outstr = outstr + ("%s = %f, " % (names[j][i], fpar[j][i]))
            initstr = initstr + "%s = %f, " % (names[j][i], tv[i])
            truestr = truestr + "%s = %f, " % (names[j][i], f[1][i])
        print(("\nTrue(%d) : %s" % (j, truestr)))
        print(("FIT(%d)   : %s" % (j, outstr)))
        print(("init(%d) : %s" % (j, initstr)))
        print(("Error:   : %f" % (Fits.fitSum2Err)))
        if func in ["exppulse", "FIGrowth1", "flattopgauss", "flattopngauss"]:
            print("red o = data; blue line is fit; black dashed is initial guess")
            pylab.figure()
            pylab.plot(numpy.array(x), yd, "ro-")
            pylab.hold(True)
            pylab.plot(numpy.array(x), initialgr, "k--")
            pylab.plot(xf[0], yf[0], "b-")  # fit
            pylab.show()


if __name__ == "__main__":
    main()
