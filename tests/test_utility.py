import pylibrary.tools.utility as U
import numpy as np
from scipy import fftpack as spFFT
from scipy import signal as spSignal

def test():

    # Clements bekkers - first generate some events
    # t = np.arange(0, 1000.0, 0.1)
    # ta = np.arange(0, 50.0, 0.1)
    # events = np.zeros(t.shape)
    # events[[50,100,250,350, 475, 525, 900, 1500, 2800, 5000, 5200, 7000, 7500],] = 1
    # tau1 = 3
    # alpha = 1.0 * (ta/tau1) * np.exp(1 - ta/tau1)
    # sig = spSignal.fftconvolve(events, alpha, mode='full')
    # sig = sig[0:len(t)]+np.random.normal(0, 0.25, len(t))
    # template = U.cb_template(shape="alpha", samplerate=1e-2, rise=0.5, decay=2.0, 
    #     ntau=2.5, lpfilter=0, predelay=0.5, dur=5)
    # (t_start, d_start) = U.clementsBekkers(sig, alpha, template=template, threshold=0.5, minpeakdist=15)
    # f = MP.figure()
    # MP.plot(t, sig, 'r-')
    # MP.plot(t, events, 'k-')
    # now call the finding routine, using the exact template (!)
    # MP.plot(t_start, d_start, 'bs')
    # MP.show()
    
    dt = 0.1
    t = np.arange(0, 100, dt)
    v = np.zeros_like(t)-60.0
    p = range(20, 900, 50)
    p1 = range(19,899,50)
    p2 = range(21,901,50)
    v[p] = 20.0
    v[p1] = 15.0
    v[p2] = -20.0
    sp = U.findspikes(t, v, 0.0, dt = dt, mode = 'schmitt', interpolate = False)
    print('findSpikes')
    print('sp: ', sp)
    # f = MP.figure(1)
    # MP.plot(t, v, 'ro-')
    si = (np.floor(sp/dt))
    print('si: ', si)
    spk = []
    for k in si:
        k = int(k)
        spk.append(np.argmax(v[k-1:k+1])+k)
    # MP.plot(sp, v[spk], 'bs')
    # MP.ylim((0, 25))
    # MP.draw()
    # MP.show()
    print("getSpikes")
    y=[]*5
    for j in range(0,1):
        d = np.zeros((5,1,len(v)))
        for k in range(0, 5):
            p = range(20*k, 500, 50 + int(50.0*(k/2.0)))
            vn = v.copy()
            vn[p] = 20.0
    d[k, 0, :] = np.array(vn) # load up the "spike" array
    y.append(d)
    tpts = range(0, len(t)) # np.arange(0, len(t)).astype(int).tolist()

def findspikes(x, v, thresh, t0=None, t1= None, dt=1.0, mode=None, interpolate=False):
    for k in range(0, len(y)):
        sp = U.getSpikes(t, y[k], 0, tpts, tdel=0, thresh=0, selection = None, interpolate = True)
        print( 'r: %d' % k, 'sp: ', sp)

#test the sine fitting routine
    from np.random import normal
    F = 1.0/8.0
    phi = 0.2
    A = 2.0
    t = np.arange(0.0, 60.0, 1.0/7.5)
# check over a range of values (is phase correct?)
    for phi in np.arange(-2.0*np.pi, 2.0*np.pi, np.pi/8.0):
        y = A * np.sin(2.*np.pi*t*F+phi) + normal(0.0, 0.5, len(t))
        (a, p) = U.sinefit(t, y, F)
        print("A: %f a: %f  phi: %f p: %f" % (A, a, phi, p))