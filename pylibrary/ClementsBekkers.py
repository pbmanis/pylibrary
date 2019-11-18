#!/usr/bin/python
"""
 Implements Clements and Bekkers algorithm for finding events (EPSCs, IPSCS) in 
 a trace.
 Clements, J., and Bekkers, Biophysical Journal, 73: 220-229, 1997.

 Incoming data is either raw or filtered data
 Rise and decay parameters determine the template that 
 is the matching template to work against the data at each point
 threshold sets the detection limit.
 The returned eventlist is a list of indices to the events detected.

 This code uses the optimizations discussed in the article.

 19 June 2002 Paul B. Manis, Ph.D. pmanis@med.unc.edu

 Modified for Python from Matlab version, July 16 and 17, 2009 Paul B. Manis.
 Pure python implementation.

"""
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as mpl


def ClementsBekkers(data, sign='+', samplerate=0.1, rise=2.0, decay=10.0, threshold=3.5,
                    dispFlag=True, subtractMode=False, direction=1,
                    lpfilter=2000, template_type=2, ntau=5,
                    markercolor=(1, 0, 0, 1)):
    """Use Clementes-Bekkers algorithm to find events in a data vector.

    The Clementes-Bekkers algorithm uses a template event shape (set by `template_type`) that is
    "swept" across the `data` set to find potential events similar to the template.

    Parameters
    ----------
    data : array
        One-dimensional array that will be searched for matching events.
    sign : str {'+', '-'}, default '+'
        Sign of events that will be searched.
    samplerate : float (default 0.1)
        the sampling rate for the data, (milliseconds)
    rise : float (default 2.0)
        The rise time of the events to be detected (milliseconds)
    decay : float (default 10.0)
        The decay time of the events to be detected (milliseconds)
    threshold : float (default 3.5)
        Threshold of event relative to stdev of data to pass detection.
    dispFlag : boolean (True)
        Controls whether or not data is displayed while searching (inactive in Python version)
    subtractMode : boolean (False)
        subtracts the best matched templates from the data array as we go. Used to 
        find overlapping events.
    direction : int (1)
        Controls direction of the search: 1 is forward from beginning, -1 is backwards from
        end of data array
    lpfilter : float (2000.)
        Low pass filter to apply to the data set before starting search. (Hz)
    template_type : int (2)
        The template type that will be used. Templates are:
            1 :  alpha function (template = ((ti - predelay) / decay) * 
                    np.exp((-(t[i] - predelay)) / decay))
            2 : exp rise to power, exp fall
            3 : exp rise to power, double exp fall
            4 : average waveform (not implemented)
            5 : 
            (see cb_template for details)
    ntau : int (5)
        number of decay time constants to use when computing template
    markerclor : (rgba) (1, 0, 0, 1)
        color of the marker used to identify matches when dispFlag is True

    Returns
    -------
    eventlist : np.array
        array of event times
    peaklist : np.array
        array of event points in the original data
    crit : numpy array
        the critierion array, a waveform of the same length as the data array
    scale : numpy array
        the scale array, a waveform of the same length as the data array
    cx : numpy array
        the cx array 
    template : numpy array
        the template used for the matching algorithm
    """   

 
    # fsamp = 1000.0/samplerate; # get sampling frequency
    # fco = 1600.0;		# cutoff frequency in Hz
    # wco = fco/(fsamp/2); # wco of 1 is for half of the sample rate, so set it like this...
    # if(wco < 1) # if wco is > 1 then this is not a filter!
    # [b, a] = butter(8, wco); # fir type filter... seems to work best, with highest order min distortion of dv/dt...
    # data = filter(b, a, data); # filter all the traces...

    # generate the template
    [template, predelay] = cb_template(funcid=template_type, samplerate=samplerate,
                                       rise=rise, decay=decay, lpfilter=lpfilter, ntau=ntau)
    N = len(template)
    if template_type is 4:  # use data
        Npost = len(template)
    else:
        Npost = int(decay * ntau / samplerate)
    isign = 1
    if sign is '-':
        isign = -1.0
 #	template = isign*template
    sumD = 0.0
    sumD2 = 0.0
    sumT = np.sum(template)  # only need to compute once.
    sumT2 = np.sum(np.multiply(template, template))
    nData = len(data)
    
    # initialize arrays used in the computation
    critwave = np.zeros(nData)  # waves for internal reference
    scalewave = np.zeros(nData)
    offsetwave = np.zeros(nData)
    cx = []
    scale = []
    pkl = []
    eventlist = []
    evn = []  # list of events
    isamp = []
    icoff = []  # cutoff
    crit = []  # criteria
    nevent = 0  # number of events
    minspacing = int(25.0 / samplerate)  # 2.0 msec minimum dt. Events cannot 
    # be closer than this direction determines whether detection is done in 
    # forward or reverse time.
    if direction == 1:
        start = 0
        finish = nData - N
    else:
        start = nData - N - 1
        finish = 0
    fN = float(N)
    lasti = start
    resetFlag = False  # force reset of running sum calculations
    # subtractMode determines whether we subtract the best fits from the data
    # as we go
    i = start
    for i in range(start, finish, direction):
        iEnd = N + i
        if i == start or resetFlag is True:
            #			print "resetting i = %d" % (i)
            sumD = np.sum(data[i:iEnd])  # optimization...
            sumD2 = np.sum(np.multiply(data[i:iEnd], data[i:iEnd]))
            ld = data[iEnd]
            fd = data[i]
            resetFlag = False
        else:  # only add or subtract the end points
            if direction == 1:
                sumD = sumD + data[iEnd] - fd
                sumD2 = sumD2 + np.multiply(data[iEnd], data[iEnd]) - (fd * fd)
                fd = data[i]
            if direction == -1:
                sumD = sumD - ld + data[i]
                sumD2 = sumD2 - (ld * ld) + np.multiply(data[i], data[i])
                ld = data[iEnd]
        sumTxD = np.sum(np.multiply(data[i:iEnd], template))
        S = (sumTxD - (sumT * sumD / fN)) / (sumT2 - (sumT * sumT / fN))
        C = (sumD - S * sumT) / fN
        # if S*isign < 0.0: # only work with correct signed matches in scaling.
        # S = 0.0   # added, pbm 7/20/09
        # f = S*template+C
        SSE = sumD2 + (S * S * sumT2) + (fN * C * C) - 2.0 * \
            (S * sumTxD + C * sumD - S * C * sumT)
        if SSE < 0:
            # needed to prevent round-off errors in above calculation
            CRITERIA = 0.0
        else:
            CRITERIA = S / np.sqrt(SSE / (fN - 1.0))
        critwave[i] = CRITERIA
        scalewave[i] = S
        offsetwave[i] = C
        # best fit to template has the wrong sign, so skip it
        if isign * S < 0.0:
            continue
        # get this peak position
        peak_pos = np.argmax(isign * data[i:iEnd]) + i
        addevent = False
        replaceevent = False
        # criteria must exceed threshold in the right direction
        if isign * CRITERIA > threshold:
            if len(eventlist) == 0:  # always add the first event
                addevent = True
            else:
                # and events that are adequately spaced
                if abs(peak_pos - pkl[-1]) > minspacing:
                    addevent = True
                else:
                    # events are close, but fit is better for this point -
                    # replace
                    if isign * CRITERIA > isign * crit[-1]:
                        replaceevent = True
        if addevent:
            eventlist.append(i)
            jEnd = iEnd
            pkl.append(peak_pos)
            crit.append(CRITERIA)
            scale.append(S)
            cx.append(C)

        if replaceevent:
            if subtractMode is True:
                j = eventlist[-1]
                jEnd = j + N
                data[j:jEnd] = data[j:jEnd] + \
                    (scale[-1] * template + cx[-1])  # add it back
            # replace last event in the list with the current event
            eventlist[-1] = i
            pkl[-1] = peak_pos
            crit[-1] = CRITERIA
            scale[-1] = S
            cx[-1] = C
        if subtractMode is True and (addevent or replaceevent):
            resetFlag = True
            # and subtract the better one
            data[i:iEnd] = data[i:iEnd] - (S * template + C)
            il = i
            i = jEnd  # restart...
        lasti = i

    nevent = len(eventlist)
    if nevent == 0:
        print('ClementsBekkers: No Events Detected')
    else:
        print('ClementsBekkers:  %d Events Detected' % (nevent))
    if dispFlag is True and nevent > 0:
        mpl.figure(1)
        t = samplerate * np.arange(0, nData)
        mpl.subplot(4, 1, 1)
        mpl.plot(t, data, 'k')
        mpl.hold(True)
        mpl.plot(t[pkl], data[pkl], marker='o',
                markerfacecolor=markercolor, linestyle='')
        mpl.plot(t[eventlist], data[eventlist], marker='s',
                markerfacecolor=markercolor, linestyle='')
        for i in range(0, len(eventlist)):
            tev = t[eventlist[i]: eventlist[i] + len(template)]
            mpl.plot(tev, cx[i] + scale[i] * template, color=markercolor)
        mpl.subplot(4, 1, 2)
        mpl.plot(t, critwave, color=markercolor)
        mpl.hold(True)
        mpl.plot([t[0], t[-1]], [threshold, threshold], 'k-')
        mpl.plot([t[0], t[-1]], [-threshold, -threshold], 'k-')
        mpl.subplot(4, 1, 3)
        mpl.plot(t, scalewave, color=markercolor, linestyle='-')
        mpl.hold(True)
        mpl.plot(t, offsetwave, color=markercolor, linestyle='--')
        tt = samplerate * np.arange(0, len(template))
        mpl.subplot(4, 2, 7)
        mpl.plot(tt, template, color=markercolor)
        mpl.draw()
    return(np.array(eventlist), np.array(pkl), np.array(crit),  
           np.array(scale), np.array(cx), np.array(template))


def testdata():
    """ Generate test data for clembek algorithm.
    
        Compute a series of EPSCs on a noisy trace to test the
        algorithm. 
    
        Parameters
        ----------
        None

        Returns
        -------
        data : numpy array
            the generated test trace
        samplerate : float
            the sample rate for the test trace
        rise : float
            rise time constant for the events
        decay : float
            decay time constant for the events
        threshold : float
            threshold value to use in the test
        sign : string (1 character)
            '-' for negative going events; '+' for positive events
    """ 
    samplerate = 0.1
    rise = 0.2
    decay = 2.0
    sign = '-'
    threshold = 2.0
    offset = -102.0
    data = np.ones(12000)*offset
    noise = np.random.randn(len(data))
    # filter the noise then put in the data array
    data = data + 0.1 * noise
    # fsamp = 1000.0/samplerate; # get sampling frequency
    # fco = 2000;		# cutoff frequency in Hz
    # wco = fco/(fsamp/2); # wco of 1 is for half of the sample rate, so set it like this...
    # if(wco < 1) # if wco is > 1 then this is not a filter!
    # fir type filter... seems to work best, with highest order min distortion of dv/dt...
    # [b, a] = butter(8, wco); 
    # data = filter(b, a, data); # filter all the traces...

    #noise2 = 6*randn(length(data), 1);
    # fco = 30;		# cutoff frequency in Hz
    # wco = fco/(fsamp/2); # wco of 1 is for half of the sample rate
    # if(wco < 1) # if wco is > 1 then this is not a filter!
    # [b, a] = butter(8, wco);
    # noise2 = filter(b, a, noise2); # filter all the traces...
    # end
    # data = data + noise2; # adding low frequency noise
    predelay = 0.0  # decay*1.0
    N = int((decay * 10.0 + predelay) / samplerate)
    template = np.zeros(N)
    t = np.arange(0, N) * samplerate
    for i in range(0, N):
        if t[i] >= predelay:
            template[i] = ((t[i] - predelay) / decay) * \
                np.exp((-(t[i] - predelay)) / decay)
    template = -template / np.amax(template)  # normalize to size 1.

    pos = 0
    i = 0  # count events generated
    tstep = int(np.floor(10000.0 / 11.0))  # interval between events
    while (pos + tstep + N) < len(data):
        pos = i * tstep
        data[pos:pos + N] = data[pos:pos + N] + template * i
        i = i + 1
    return(data, samplerate, rise, decay, threshold, sign)


def cb_template(funcid=1, samplerate=0.1, rise=2.0, decay=10.0, lpfilter=2000.0,
                ntau=5,  mindur=50.0):
    """ Compute the template waveform

    Compute one of several possible template waveforms chosen by `funcid`, possibly 
    low pass filtered as set by `lpfilter`, to be used in the Clements-Bekkers algorithm.

    Parameters
    ----------
    funcid : int (1)
        Function selector:
            1 = alpha
            2 = exp^2 rise, exp decay
            3 = like 2, but different (aaargh...)
    samplerate : float (0.1)
        Samplerate for the template - should be the same as the data
    rise : float (2.0)
        Rise time parameter of the template, in msec
    decay : float (10.0)
        Decay time parameter of the template, in msec
    lpfilter : float (2000.)
        Low pass filter applied to the template, corner frequency in Hz.
    ntau : int (5)
        Duration of the template waveform based on the decay time constant
    mindur : float (50.0)
        Minimum duration of the template, in msec.

    Returns
    -------
    template : array (numpy)
        The template waveform for use by ClementsBekkers
    predelay : float
        the predelay time for the waveform (baseline); usually 0.

    """
    predelay = 0.0  # 3*rise
    dur = predelay + ntau * decay
    if dur < mindur:
        dur = mindur
    N = int(dur / samplerate)
    template = np.zeros(N)
    t = samplerate * np.arange(0, N)
    if funcid == 1:
        for i in range(0, N):
            if t[i] >= predelay:
                template[i] = ((t[i] - predelay) / decay) * \
                    np.exp((-(t[i] - predelay)) / decay)
        template = template / np.amax(template)  # normalized to size 1.
        return(template, predelay)

    if funcid == 2:  # standard for EPSC detection
        for i in range(0, N):
            if t[i] >= predelay:
                template[i] = (
                    1.0 - np.exp(-(t[i] - predelay) / rise)) ** 2.0 * np.exp(-(t[i] - predelay) / decay)
#    if(lpfilter > 0)
# fsamp = 1000/samplerate; # get sampling frequency
# fco = lpfilter;		# cutoff frequency in Hz
# wco = fco/(fsamp/2); # wco of 1 is for half of the sample rate, so set it like this...
# if(wco < 1) # if wco is > 1 then this is not a filter!
#            [b, a] = butter(8, wco);
# template = filter(b, a, template); # filter each trace...
#        end;
#    end;
#
        m = np.amax(template)
        template = template / m  # normalize to size 1.
        return(template, predelay)

    if funcid == 3:  # dual exponential function with power
        for i in range(0, N):
            if t[i] >= predelay:
                td = t[i] - predelay
                template[i] = (
                    (1.0 - np.exp((-td / rise))) ** 2.0) * np.exp(-td / decay)
        m = np.amax(template)
        template = template / m  # normalize to size 1.
        return(template, predelay)

    if funcid == 4:  # use data from average waveform somewhere
        pass
    # 5/13/2008.
    # aveargin/zoom, etc put the displayed data into the "stack" variable
    # this is then usable as a template. It is normalized first, and
    # sent to the routine.
    #    predelay = 0.05;
    # template = STACK{1}; # pull from the new "stack" variable.
    #    template = template/max(template);
    if funcid == 5:  # use the average of the prior analysis.
        pass
        #    sf = getmainselection();
    #    predelay = CONTROL(sf).EPSC.predelay;
    #    template = CONTROL(sf).EPSC.iavg;
    #    template = template/max(template);
    #
    # otherwise
    #    template = [];
    #    end;


def cb_multi(data, sign='+', samplerate=0.1, rise=[5.0, 3.0, 2.0, 0.5], decay=[30.0, 9.0, 5.0, 1.0],
                        matchflag=True, threshold=3.0,
                        dispflag=True, lpfilter=0, template_type=1, ntau=5):
    
    """ return the best choice of fits among a set of rise and fall times using the CB method.
            the tested rise and decay times are determined by the lists of rise and decay arrays.
            if matchflag is true, then the arrays are compared in order (rise[0], decay[0]),
            (rise[1], decay[1]). if matchflag is false, then all pairwise comparisons are made

    """

    clist = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1),
             (1, 0.5, 0, 5, 1), (0.5, 1, 0.5, 1), (0.5, 0.5, 1, 1), (0, 0, 0, 1)]

    nrTests = 1
    ndTests = 1
    if len(rise) > 1:
        nrTests = len(rise)
    if len(decay) > 1:
        ndTests = len(decay)
    nTests = max((nrTests, ndTests))
    if matchflag is False:
        nCand = nrTests * ndTests
    else:
        nCand = nrTests
    icand = np.array([])
    iscamp = np.array([])
    ioff = np.array([])
    peaks = np.array([])
    crit = np.array([])
    itj = np.array([])
    itk = np.array([])
    # datan will be modified during CB if subtractMode is on
    datan = data.copy()
    for k in range(0, nrTests):  # use multiple template shapes
        for j in range(0, ndTests):
            if matchflag is True and j != k:
                continue
            (ic, pks, critval, isc, io, template) = ClementsBekkers(datan,
                                                                    samplerate=samplerate, rise=rise[k], decay=decay[
                                                                        j], threshold=threshold, sign=sign,
                                                                    dispFlag=dispflag, subtractMode=True,
                                                                    lpfilter=lpfilter, template_type=template_type, ntau=ntau, markercolor=clist[k])
#	returns :: (eventlist, pkl, crit, scale, cx)
            if ic is []:
                continue
            icand = np.append(icand, ic)
            peaks = np.append(peaks, pks)
            crit = np.append(crit, critval)
            iscamp = np.append(iscamp, isc)
            ioff = np.append(ioff, io)
            itj = np.append(itj, j * np.ones(len(ic)))
            itk = np.append(itk, k * np.ones(len(ic)))

    dist = 10.0  # minimum time bewteen events is set to 5 msec here.
    # pairwise comparision
    if sign is '-':
        ksign = -1
    else:
        ksign = 1
    print(np.shape(icand))
    nt = len(icand)
    if nt is 0:
        return
    # choose the best fit candidate events within dist of each other
    for ic in range(0, nt):
        # compare each candidate template with the others
        for jc in range(ic + 1, nt):
            if icand[jc] is -1 or icand[ic] is -1:
                continue
            if abs(icand[ic] - icand[jc]) < dist or abs(peaks[ic] - peaks[jc]) < dist:
                if ksign * crit[ic] > ksign * crit[jc]:
                    icand[jc] = -1  # removes an event from the list
                else:
                    icand[ic] = -1
    mcand = ma.masked_less(icand, 0)
    selmask = ma.getmask(mcand)
    icand = ma.compressed(mcand)
    crit = ma.compressed(ma.array(crit, mask=selmask))
    peaks = ma.compressed(ma.array(peaks, mask=selmask))
    iscamp = ma.compressed(ma.array(iscamp, mask=selmask))
    ioff = ma.compressed(ma.array(ioff, mask=selmask))
    itj = ma.compressed(ma.array(itj, mask=selmask))
    itk = ma.compressed(ma.array(itk, mask=selmask))
    mpl.figure(2)
    t = samplerate * np.arange(0, len(data))
    mpl.subplot(1, 1, 1)
    mpl.plot(t, data, 'k', zorder=0)
    mpl.hold(True)
    ipts = icand.astype(int).tolist()
    ippts = peaks.astype(int).tolist()
    ijp = itj.astype(int).tolist()
    cols = []
    for p in range(0, len(ippts)):
        cols.append(clist[ijp[p]])  # plots below were t[ipts], data[ipts]
    mpl.scatter(t[ipts], ioff, s=49, c=cols, marker='s', zorder=1)
    mpl.scatter(t[ippts], iscamp, s=49, c=cols, marker='o', zorder=2)
    mpl.show()

    return(icand, peaks, crit, iscamp, ioff)


def test():
    """ Provide a test of the Clements-Bekkers algorithm
    """
    (data, samplerate, rise, decay, threshold, sign) = testdata()
    ClementsBekkers(data, dispFlag=True, samplerate=samplerate,
                    subtractMode=True, direction=-1,
                    template_type=1, rise=rise, decay=decay, threshold=4.0,
                    ntau=8, sign=sign)
    mpl.show()

if __name__ == "__main__":
    test()

##	cb_multi(data, sign=sign, matchflag = True,  dispflag = True, threshold = 4.5)
