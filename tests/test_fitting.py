import matplotlib.pyplot as pylab
import pylibrary.fitting.fitting as fitting
import numpy as np
import pytest

def compare_vals(funcname, names, fpar, f):
    j = 0
    tol = 1e-4
    if funcname in ["FIGrowth1", "poly3", "poly4"]:
        return
    if funcname in ["taucurve"]:
        tol = 1e-3

    for i in range(0, len(names[j])):
        print(
            f"Names, values: , {names[j][i]:s} = {fpar[j][i]:.9f}, {f[1][i]:.9f}  ")
        if f[1][i] != 0.0:
            assert fpar[j][i] == pytest.approx(f[1][i], rel=tol)
        else:
            if np.abs(fpar[j][i] - f[1][i]) > 1e-4:
                print(f"Error: {names[j][i]} = {fpar[j][i]:f}, {f[1][i]:f}")
                assert False
    # print((    "Error:   : %f" % (Fits.fitSum2Err)))

def test_fitting():
    print("Testing Fitting")
    Fits = fitting.Fitting()

    exploreError = False

    func = "flattopngauss"
    if exploreError is True:
        # explore the error surface for a function:

        f = Fits.fitfuncmap[func]
        p1range = np.arange(0.1, 5.0, 0.1)
        p2range = np.arange(0.1, 5.0, 0.1)

        err = np.zeros((len(p1range), len(p2range)))
        x = np.array(np.arange(f[4][0], f[4][1], f[4][2]))
        C = None
        if func in ["expsum2", "exppulse", "ngauss", "normalizedgauss"]:
            C = f[7]
            tv = f[5]
        # check exchange of tau1 ([1]) and width[4]
        yOffset, t0, tau1, tau2, amp, width = f[1]  # get inital parameters
        y0 = f[0](f[1], x, C=C)
        noise = np.random.random(y0.shape) - 0.5
        y0 += 0.0 * noise
        sh = err.shape
        yp = np.zeros((sh[0], sh[1], len(y0)))
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
        # if func != 'flattopngauss':
        #     continue
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
        x = np.array(np.arange(f[4][0], f[4][1], f[4][2]))
        C = None
        if func in ["expsum2", "exppulse", "ngauss", "flattopgauss", "flattopngauss"]:
            print("\n", f[7])
            C = f[7]
            tv = f[5]

        y = f[0](f[1], x, C=C)
        yd = np.array(y)
        noise = np.random.normal(0, 0.1, yd.shape)
        my = np.amax(yd)
        # yd = yd + sigmax*0.05*my*(np.random.random_sample(shape(yd))-0.5)
        yd += noise * my / signal_to_noise
        testMethod = "SLSQP"
        if func == "taucurve":
            # continue
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
                np.array([1]), 0, x, yd, fitFunc=func, bounds=bounds, method=testMethod
            )
        elif func == "FIGrowth1":
            # Parameter p is [Fzero, Ibreak, F1amp, F2amp, Irate]
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
                np.array([1]),
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
                np.array([1]), 0, x, yd, fitFunc=func, bounds=bounds, method=testMethod
            )
            tv = f[5]

        elif func == "exp2":
            bounds = [(-0.001, 0.001), (-5.0, 0.0), (1.0, 500.0), (-5.0, 0.0), (1.0, 10000.0)]
            (fpar, xf, yf, names) = Fits.FitRegion(
                np.array([1]), 0, x, yd, fitFunc=func, bounds=bounds, method=testMethod
            )
            tv = f[5]

        elif func == "exppulse":
            # set some constraints to the fitting
            # yOffset, tau1, tau2, amp, width = f[1]  # order of constraings
            dt = np.mean(np.diff(x))
            xd = x.copy()
            bounds = [(-5, 5), (-15.0, 15.0), (-2, 2.0), (2 - 10, 10.0), (-5, 5.0), (0.0, 5.0)]
            # example for constraints:
            # cons = (
            #     {"type": "ineq", "fun": lambda xa: xa[4] - 3.0 * xa[2]},
            #     {"type": "ineq", "fun": lambda xa: -xa[4] + 12 * xa[2]},
            #     {"type": "ineq", "fun": lambda xa: xa[2]},
            #     {"type": "ineq", "fun": lambda xa: -xa[4] + 2000},
            # )

            # cons = ({'type': 'ineq', 'fun': lambda x: x[3] - x[2]},  # tau1 < tau2
                    # )
            C = None

            tv = f[5]
            initialgr = f[0](f[5], x, None)
            (fpar, xf, yf, names) = Fits.FitRegion(
                np.array([1]),
                0,
                xd,
                yd,
                fitFunc=func,
                fixedPars=C,
                # constraints=cons,
                bounds=bounds,
                method=testMethod,
            )


        # elif func in ["gauss", "normalizedgauss"]:
        #     continue
        else:
            tv = f[5]

            initialgr = f[0](f[5], x, C=C)
            (fpar, xf, yf, names) = Fits.FitRegion(
                np.array([1]),
                0,
                x,
                yd,
                fitFunc=func,
                fixedPars=C,
                constraints=cons,
                bounds=bnds,
                method=testMethod,
            )
            # print("names: ", names)
            # print("tv: ", tv)
            # print fpar
        
        s = np.shape(fpar)

        # j = 0
        # outstr = ""
        # initstr = ""
        # truestr = ""
        # for i in range(0, len(names[j])):
        #     # print(i, names, len(names[j]), tv)
        #     #            print "%f " % fpar[j][i],
        #     outstr = outstr + ("%s = %f, " % (names[j][i], fpar[j][i]))
        #     initstr = initstr + "%s = %f, " % (names[j][i], tv[i])
        #     truestr = truestr + "%s = %f, " % (names[j][i], f[1][i])
        # print(("\n    True(%d) : %s" % (j, truestr)))
        # print(("    FIT(%d)   : %s" % (j, outstr)))
        # print(("    init(%d) : %s" % (j, initstr)))
        # print((    "Error:   : %f" % (Fits.fitSum2Err)))
        compare_vals(funcname = func, names=names, fpar=fpar, f=f)

