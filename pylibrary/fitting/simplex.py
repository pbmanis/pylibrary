"""
#   ------------------------------------------------------------

    Modified version of simp from byte march 1984
    for the DATAC program.
    The parameters are passed into var[] and the mean result
    is returned into it.

    The theoretical equation and the data are taken from a strip

    Update 18-11-1985   D.B. replaced pow( ,2.) by *
    Important Update for the data entry and display
    September 1986 D.Bertrand

    Update 11/22/88 - Let blanking on strip be used to delete points
    from the lms computation in the fit (P. Manis )

    Modified to work with getcmd.c and MSC V5.0 P. Manis, 1/90.

    Update 24-Apr-90 Added weighting function to lms fit -  you specify strip.

    tested 2014: does not work correctly. 
    
    -------------------------------------------------------------

   Converted ("translated") to pure python version, 12/29/2009
   Paul B. Manis, as a stand-alone program.
   verified to work on exp function (test by running "simplex.py")

Copyright 2010-2014  Paul Manis
Distributed under MIT/X11 license. See license.txt for more infofmation.

"""
import numpy as np
import numpy.random as random
import sys

#from PyQt4 import Qt, QtCore, QtGui
#import PyQt4.Qwt5 as Qwt
#from PyQt4.Qwt5.anynp import *
#from lib.qtgraph.graphicsItems import *
#import lib.qtgraph.widgets
#from lib.qtgraph.GraphicsView import *
#import lib.util.PlotWidget
#import lib.util



NVP = 2   #total number of variables per data pts
ALFA = 0.1 # reflection coeficient > 0
BETA = 0.5 # contraction coeficient 0 to 1
GAMMA = 2.0# expansion coeficient >1
ROOT2 = np.sqrt(2)
N = 2

dx = []
dy = []
vb = []  # the blanking array with special mode (0 to mask, 1 to leave intact)
lb = []
hb = []
pars = []
simp = []
mean = []
l = []
h = []
curve = []
curve2 = []
curve3 = []
pw = []

watchit = False
npts = 0
mask_flag = 0   # mask flag
done = 0  # end iteration flag
ndisp = 10  # number for the display
nrnd = 0

ssfirst = 0  # 0 until run once

quiet = True

skipcnt = 1  # skip counter: calcuate rms at skipped points
ncount = 1
npx = 0
maxiter = 5000
niter = 1
maxer = 0.0


def sum_res(func, x):
    """  compute the residual

    The residual is sum of square of difference between equation/model and data,
    after clipping data x to the bounds


    """
    global lb, hb, dx, dy, quiet
    # impose limits on x[i] for acceptable values for computation, if necessary

    for i in range(0, m):
        if(x[i] < lb[i]):
            x[i] = lb[i]    # also modify values for computation
        if(x[i] > hb[i]):
            x[i] = hb[i]
    # compute the equation - run silently unless there's an error!
    (vr, dt) = func(x, dx)
    lvr = len(vr)
    ldy = len(dy)
    dyt = dy
    vrt = vr
    if lvr < ldy:
        dyt = np.array(dy[0:lvr])
    if lvr > ldy:

        vrt=np.array(vr[0:ldy])
    d = (vrt - dyt)**2
#   if(mask_flag > 0) vxv_s(vt, vt, vb, npts)  # zero the masked points so we don't use them in the
    x[m] = np.sum(d) # sum of squared difference is total lms error
    if not quiet:
        print ("Sumres\nPars", x)
        print( "Error: %f " % (x[m]))


def order():
    global simp, n, l, h
    for j in range(0, n):  # for all vertex
        for i in range(0, n):  # of all vertex find the best and the worst
            if(simp[i][j] < simp[l[j]][j]):
                l[j] = i
            if(simp[i][j] > simp[h[j]][j]):
                h[j] = i
    return
# try this:
# for i in range(0, n): # of all vertex find the best and the worst
#            if(simp[i][m] < simp[l[m]][m]):
#                l[m]=i
#            if(simp[i][m] > simp[h[m]][m]):
#                h[m]=i


def first():
    """ print starting values of the simplex function
    """
    global n, m, simp, ncount
    print ("Starting simplex")
    ncount = 1
    if not quiet:
        for j in range(0, n):
            print( "Simp [%2d]  " % (j))
            for i in range(0, n):
                print(f"{simp[j][i]:9.3g}  ")


def new_vertex():
    """ Create a new vertex for the simplex
    """
    global niter, ndisp, n, m, anext, lb, hb, simp, error, quiet
    if not np.mod(niter, ndisp) and not quiet:
       print( "-- New Vertex for Simplex Iteration %5d --" % (niter))
    for i in range(0, n):
        if(i < n-1) :
            if(anext[i] < lb[i]):
                anext[i] = lb[i]
            if(anext[i] > hb[i]):
                anext[i] = hb[i]

        simp[h[m]][i] = anext[i]
        if(~(niter % ndisp)) and not quiet:
            if(i < (n-1)):
                print ("n[%1d]    %8.3g    %8.3e" %(i, anext[i], error[i]))
            else:
                print ("lms:    %8.3g" % (anext[n-1])) # for that last one


def report(func):
    """ report the final fit results
    """
    global mean, m, n, simp, error, niter, quiet, npx
    i = j = 0

    print( "=====================================\n\nEnd of simplex iterations")
    print( "Total iterations: %d" % niter)
    print ("Summary of results:")
    for j in range(0, n):
        print ("\nSimp [%2d]  " % (j),)
        for i in range(0, n):
            print ("%7.3g  " % (simp[j][i]),)
    print( " ")

    print ("The final mean values & errors are: ")
    for i in range(0, n - 1):
        # final values
        print ("n[%1d] %9.3g     frac err: %9.3g" % (i, mean[i], error[i]))
    sum_res(func, mean)  # compute it with final parameters
    sigma = mean[m]

    sigma = np.sqrt((sigma/float(npx)))
    print( "The standard deviation is:  %9.3g" % (sigma))
    if(npx >  m):
        sigma = sigma/np.sqrt(float(npx - m))
    else:
        sigma = sigma/np.sqrt(float(m-npx))
    print( "The estimated error is:     %9.3g" % (sigma))


def simplex(func, ipars, X, Y, lbound=None, hbound=None, maxiters=1000,
       silent=False, watch=False):
    """ Perform the simplex, after initializing parameters
    """
    global dx, dy, N, m, n, p, q, step, center, anext, mean, error, maxerr, ssfirst, recalcflag
    global lb, hb, simp, maxiter, quiet, npx
    global curve, curve2, pw, watchit
    maxiter = maxiters
    watchit = watch
    quiet = silent
    N = len(ipars) + 1
    n = len(ipars) + 1
    m = len(ipars)
    dx = X
    dy = Y
    npx = len(dx)
    pars = ipars
# save_flag=recalc_flag# save the actual flag
    i = j = 0
    recalc_flag = 0# clear the recalc flag during simplex
    if ssfirst == 0:        # make sure variables are initialized
        if lbound is None:
            lb = [-np.inf] * N
        else:
            lb = lbound
        if hbound is None:
            hb = [np.inf] * N
        else:
            hb = hbound
        anext = np.zeros(N)
        mean = center = error = maxerr = p = q  = anext
        step = np.ones(N)*0.1
        for i in range(0, len(ipars)):
            if abs(pars[i]) < 1e-3:
                step[i] = 1e-3

        simp = np.zeros((N,N))
        simp[0,0:N-1] = ipars

        ssfirst = 1
    # if watchit:
    #    win = Qt.QMainWindow()
    #    pw = lib.util.PlotWidget.PlotWidget()
    #    win.setCentralWidget(pw)
    #    win.show()
    #    pw.enableAutoScale()
    #    curve = pw.plot(data=Y, x=X)
    #    curve.setPen(Qt.Qt.blue)
    #    (yp, dt) = func(X, ipars)
    #    curve2 = pw.plot(data=yp, x=X)
    #    pw.attachCurve(curve)
    #    pw.attachCurve(curve2)
    #    pw.replot()
        # pw.setLabel('left', "Y Mag")
        # pw.setLabel('bottom', "X Time")
    _simplex_fit(func)
    return(simp[0])


def _simplex_fit(func):
    """ The simplex fit function 
    """
    global dx, dy, N, m, n, p, q, step, center, anext, mean, error, maxerr, ssfirst, recalcflag
    global lb, hb, simp, l, h, niter, maxiter
    global curve, curve2, curve3, pw, watchit
    npts = len(dy)
    sum_res(func, simp[0])  # first vertex is the initial guess
    if not quiet:
        print ('simp: ', simp)
        print( 'Initial ', simp)
        print( 'step ', step)
    #   maxerr[m] = 0.0

    for i in range(0, m): #  compute the offset of the vertex of the starting simplex
        p[i] = step[i] * (np.sqrt(float(n)) + float(m) -  1.0) / (float(m)*ROOT2) #* (np.random.randn()+1)
        q[i] = step[i] * (np.sqrt(float(n)) - 1.0) / (float(m)*ROOT2) #* (np.random.randn()+1)
    if not quiet:
        print( 'P ', p)
        print( 'Q ', q)
    nerr_condx = 0
    for i in range(0, n):
        if maxerr[i] > 0.0:
            nerr_condx += 1
    for i in range(1, n):  # all vertex of the starting simplex
        for j in range(0, m):
            simp[i][j] = simp[0][j] + q[j]
        simp[i][i - 1] = simp[0][i - 1] + p[i - 1]
        sum_res(func, simp[i])  # and their residuals
    if not quiet:
        print( 'simp: ', simp)
    # preset
    l = [1] * N
    h = [1] * N
    order()
    first()
    #   ----- iteration loop ------
    for niter in range(0, maxiter):
        if not quiet:
            print ("Starting Iteration: %d" % (niter))
        center = np.zeros(n)
        nc = np.zeros(n)
        for i in range(0, n):  # compute the centroid
            if i is not h[m]:  # excluding the worst
                for  j in range(0, m):
                    center[i] += simp[i][j]  # sums here
                    nc[j] = nc[j] + 1
        center = center/float(n-1)
        for i in range(0, n):  # first attempt to reflect
            anext[i] = simp[h[m]][i]+(1.0+ALFA)*(simp[h[m]][i]-center[i])
            # new vertex is the specular reflection of the worst
        sum_res(func, anext)
        if anext[m] <= simp[l[m]][m]: # better than the best ?
            (yp, dt) = func(dx, simp[l[m]][:])
#            if watchit:
#                curve3 = pw.plot(data= yp, x=dx)
#                curve3.setPen(Qt.Qt.yellow)
#                pw.attachCurve(curve3)
#                pw.replot()
            new_vertex()  #   accepted
            for i in range(0, m):
                anext[i]= GAMMA * simp[h[m]][i]+(1.0-GAMMA) * center[i]
            sum_res(func, anext) # still better ?
            if anext[m] <= simp[l[m]][m]:
                new_vertex()
        # expansion accepted
        else:  # if not better than the best
            if anext[m] <= simp[h[m]][m]:
                new_vertex()  # better, worst
            else:   # worst than worst - contract
                for i in range(0, m):
                    anext[i] = BETA * (simp[h[m]][i]+ center[i])
                sum_res(func, anext)
                if anext[m] <= simp[h[m]][m]:
                    new_vertex()  # contraction ok
                else:  # if still bad shrink all bad vertex
                    for i in range(0, n):
                        for j in range(0, m):
                            simp[i][j] = BETA * (simp[i][j] + simp[l[m]][j])
                        sum_res(func, simp[i])

        order()
        done = 0
        for j in range(0, n):
            if simp[h[j]][j] == 0.0:
                error[j] = 1.0  # avoid division by 0
            else:
                error[j]= abs((simp[h[j]][j] - simp[l[j]][j])/simp[h[j]][j])
            if maxerr[j] > 0.0 and (maxerr[j] > error[j]) :
                done += 1  # when we get below a particular limit, we set the flag

#       print(20,"Ncount: %d",ncount++)
        if((nerr_condx > 0) and (done == nerr_condx)):
            break # all set limits have been reached
    mean = np.zeros(n)
    for i in range(0, n): # average each parameter
        for j in range(0, n):
            mean[i] += simp[j][i]
        mean[i] = mean[i] / float(n)

    report(func)


def testf(p, x, noise=0.):
    x = np.array(x)
    y = p[0]*(1.0 - np.exp(-x/p[1]))*np.exp(-x/p[2])+p[3]
    if noise > 0.:
        y = y + np.random.normal(loc=0., scale=noise, size=y.shape[0])
    return (np.array(y), np.mean(np.diff(x)))

def main():
     """ self test """
 #    import matplotlib.pylab as MP
     #app = QtGui.QApplication(sys.argv)

     x = np.arange(0, 50, 0.01);
     p0 = [2, 1, 3., -60]
     (y, dt) = testf(p0, x, noise=0.05)
     pnew = simplex(testf, p0, x, y, lbound=[0, 0.01, 0.01, -100], hbound=[10, 10, 10, 10],
                    silent=True, watch=False, maxiters=1000)
     print ('final parmaters: ', pnew)
     import matplotlib.pylab as MP
     MP.figure()
     MP.plot(x, y, 'k')  # original functoin
     #(ys, dt) = testf(p0, x)
     #MP.plot(x, ys, 'g', linewidth=2)  # initial guess
     (yf, dt) = testf(pnew, x)
     MP.plot(x, yf, 'r--', linewidth=2.0)  # converged solution
     MP.show()   
    
if __name__ == "__main__":
    main()
