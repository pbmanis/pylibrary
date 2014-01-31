#!/usr/bin/python
"""
(evn, isamp, icoff, crit, template, predelay) = ClementsBekkers(data, samplerate, rise, decay, threshold, sign, dispflag, lpfilter, template_type, ntau)
 implement Clements and Bekkers algorithm
 Biophysical Journal, 73: 220-229, 1997.

 incoming data is raw data (or filtered)
 rise and decay determine the template that 
 is the matching template to work against the data at each point
 threshold sets the detection limit.
 the returned eventlist is a list of indices to the events detected.

 This code uses the optimizations discussed in the article.

 19 June 2002 Paul B. Manis, Ph.D. pmanis@med.unc.edu

 Modified for Python from Matlab versoin, July 16 and 17, 2009 Paul B. Manis.

"""
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as pl

def ClementsBekkers(data, sign='+',samplerate=0.1, rise=2, decay=10, threshold=3.5, 
					dispFlag=True, subtractMode = False, direction = 1,
					lpfilter = 2000, template_type = 2, ntau=5,
					markercolor = (1,0,0,1)):
	evn = []
	isamp = []
	icoff = []
	crit = []
	nevent = 0

#        fsamp = 1000.0/samplerate; # get sampling frequency
#        fco = 1600.0;		# cutoff frequency in Hz
#        wco = fco/(fsamp/2); # wco of 1 is for half of the sample rate, so set it like this...
#        if(wco < 1) # if wco is > 1 then this is not a filter!
#            [b, a] = butter(8, wco); # fir type filter... seems to work best, with highest order min distortion of dv/dt...
#            data = filter(b, a, data); # filter all the traces...

	[template, predelay] = cb_template(funcid=template_type, samplerate=samplerate,
					rise=rise, decay=decay, lpfilter = lpfilter, ntau=ntau)
	N = len(template);
	if template_type is 4: # data
		Npost = len(template)
	else:
		Npost = int(decay*ntau/samplerate)
	if sign is '-':
		isign = -1.0
	else:
		isign = 1.0
#	template = isign*template
	sumD=0.0
	sumD2=0.0
	sumT = np.sum(template) # only need to compute once.
	sumT2 = np.sum(np.multiply(template,template))
	nData = len(data)
	critwave = np.zeros(nData) # waves for internal reference
	scalewave = np.zeros(nData)
	offsetwave = np.zeros(nData)
	crit=[] # lists of result
	cx=[]
	scale=[]
	pkl=[]
	eventlist=[]
	minspacing = int(25.0/samplerate) # 2.0 msec minimum dt
	# Measurement is done in C code in clembek. The following is just adapted from the matlab
	# originaly used to develop the C code. 
	# direction determines whether detection is done in forward or reverse time.
	if direction == 1:
		start = 0
		finish = nData-N
	else:
		start = nData-N-1
		finish = 0
	fN = float(N)
	lasti = start
	resetFlag = False # force reset of running sum calculations
	#subtractMode determines whether we subtract the best fits from the data as we go
	i = start
	for i in range(start, finish, direction):
		iEnd = N+i
		if i == start or resetFlag is True:
#			print "resetting i = %d" % (i)
			sumD = np.sum(data[i:iEnd]) # optimization...
			sumD2 = np.sum(np.multiply(data[i:iEnd],data[i:iEnd]))
			ld = data[iEnd]
			fd = data[i]
			resetFlag = False
		else: # only add or subtract the end points
			if direction == 1:
				sumD = sumD + data[iEnd] - fd
				sumD2 = sumD2 + np.multiply(data[iEnd],data[iEnd]) - (fd*fd)
				fd = data[i]
			if direction == -1:
				sumD = sumD - ld + data[i]
				sumD2 = sumD2 - (ld*ld) + np.multiply(data[i],data[i])
				ld = data[iEnd]
		sumTxD = np.sum(np.multiply(data[i:iEnd],template))
		S = (sumTxD - (sumT * sumD/fN))/(sumT2 - (sumT * sumT/fN))
		C = (sumD - S*sumT)/fN
#		if S*isign < 0.0: # only work with correct signed matches in scaling.
#			S = 0.0   # added, pbm 7/20/09
#		f = S*template+C
		SSE = sumD2 + (S*S*sumT2) + (fN*C*C) - 2.0 * (S*sumTxD + C*sumD - S*C*sumT)
		if SSE < 0:
			CRITERIA = 0.0 # needed to prevent round-off errors in above calculation
		else:
			CRITERIA = S/np.sqrt(SSE/(fN-1.0))
		critwave[i] = CRITERIA
		scalewave[i] = S
		offsetwave[i] = C
		if isign*S < 0.0: # best fit to template has the wrong sign, so skip it
			continue
		peak_pos = np.argmax(isign*data[i:iEnd])+i # get this peak position
		addevent = False
		replaceevent = False
		if isign*CRITERIA > threshold:	# criteria must exceed threshold in the right direction
			if len(eventlist) == 0: # always add the first event
				addevent = True
			else:
				if abs(peak_pos - pkl[-1]) > minspacing: # and events that are adequately spaced
					addevent = True
				else:
					if isign*CRITERIA > isign*crit[-1]: # events are close, but fit is better for this point - replace
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
				data[j:jEnd] = data[j:jEnd] + (scale[-1]*template + cx[-1]) # add it back
			eventlist[-1]=i # replace last event in the list with the current event
			pkl[-1]=peak_pos
			crit[-1]=CRITERIA
			scale[-1]=S
			cx[-1]=C
		if subtractMode is True and (addevent or replaceevent):
			resetFlag = True
			data[i:iEnd] = data[i:iEnd] - (S*template + C) # and subtract the better one
			il = i
			i = jEnd # restart...
#			print 'restart @ i = %d (il = %d)' % (i, il)
#			print addevent
#			print replaceevent
			
		lasti = i

		
	nevent = len(eventlist)
	if nevent == 0:
		print 'ClementsBekkers: No Events Detected'
	else:
		print 'ClementsBekkers:  %d Events Detected' % (nevent)
#	print dispFlag
#	print nevent
	if dispFlag is True and nevent > 0:
		pl.figure(1)
		t = samplerate*np.arange(0,nData)
		pl.subplot(4,1,1);
		pl.plot(t, data, 'k');
		pl.hold(True)
		pl.plot(t[pkl], data[pkl], marker = 'o', markerfacecolor=markercolor, linestyle='' );
		pl.plot(t[eventlist], data[eventlist], marker = 's', markerfacecolor=markercolor, linestyle='');
		for i in range(0, len(eventlist)):
			tev = t[eventlist[i]: eventlist[i]+len(template)]
			pl.plot(tev, cx[i]+scale[i]*template, color=markercolor)
		pl.subplot(4,1,2);
		pl.plot(t, critwave, color=markercolor);
		pl.hold(True)
		pl.plot([t[0],t[-1]],[threshold,threshold], 'k-');
		pl.plot([t[0],t[-1]],[-threshold,-threshold], 'k-');
		pl.subplot(4,1,3);
		pl.plot(t, scalewave, color=markercolor, linestyle='-');
		pl.hold(True)
		pl.plot(t, offsetwave, color=markercolor, linestyle='--' );
		tt = samplerate*np.arange(0, len(template))
		pl.subplot(4,2,7)
		pl.plot(tt, template, color=markercolor)
		pl.draw()
	return(np.array(eventlist), np.array(pkl), np.array(crit),  np.array(scale), np.array(cx),
		   np.array(template))
		
def testdata():
# returns [data, samplerate, rise, decay, threshold, sign, dispflag]
	samplerate = 0.1
	rise = 3.0
	decay = 9.0
	sign = '-'
	threshold = 2.0
	data = np.zeros(12000)
	noise = np.random.randn(len(data));
	# filter the noise then put in the data array
	data = data + 0.*noise
	#fsamp = 1000.0/samplerate; # get sampling frequency
	#fco = 2000;		# cutoff frequency in Hz
	#wco = fco/(fsamp/2); # wco of 1 is for half of the sample rate, so set it like this...
	#if(wco < 1) # if wco is > 1 then this is not a filter!
	#	[b, a] = butter(8, wco); # fir type filter... seems to work best, with highest order min distortion of dv/dt...
	#	data = filter(b, a, data); # filter all the traces...

	#noise2 = 6*randn(length(data), 1);
	#fco = 30;		# cutoff frequency in Hz
	#wco = fco/(fsamp/2); # wco of 1 is for half of the sample rate, so set it like this...
	#if(wco < 1) # if wco is > 1 then this is not a filter!
	#   [b, a] = butter(8, wco); # fir type filter... seems to work best, with highest order min distortion of dv/dt...
	#   noise2 = filter(b, a, noise2); # filter all the traces...
	#end
	#data = data + noise2; # adding low frequency noise
	predelay = 0.0 # decay*1.0
	N = int((decay*10.0+predelay)/samplerate)
	template = np.zeros(N)
	t = np.arange(0, N)*samplerate
	for i in range(0,N):
		if t[i] >= predelay:
				template[i] = ((t[i] - predelay)/decay)*np.exp((-(t[i] - predelay))/decay)
	template = -template/np.amax(template) # normalize to size 1.
	
	pos = 0
	i = 0 # count events generated
	tstep = int(np.floor(10000.0/11.0)) # interval between events
	while (pos+tstep+N) < len(data):
		pos = i * tstep
		data[pos:pos+N] = data[pos:pos+N]+template*i
		i = i + 1
	return(data, samplerate, rise, decay, threshold, sign)

def cb_template(funcid=1, samplerate=0.1, rise=2.0, decay=10.0, lpfilter=2000.0,
				ntau=5, wave=None, mindur = 50.0):
	""" returns template (a numpy array), predelay """
	predelay = 0.0 #  3*rise
	dur = predelay+ ntau*decay
	if dur < mindur:
		dur = mindur
	N = int(dur/samplerate)
	template=np.zeros(N)
	t = samplerate*np.arange(0, N)
	if funcid == 1:
		for i in range(0,N):
			if t[i] >= predelay:
				template[i] = ((t[i] - predelay)/decay)*np.exp((-(t[i] - predelay))/decay)
		template = template/np.amax(template) # normalized to size 1.
		return(template, predelay)
		
	if funcid == 2: # standard for EPSC detection
		for i in range(0,N):
			if t[i] >= predelay:
				template[i] = (1.0-np.exp(-(t[i]-predelay)/rise))**2.0*np.exp(-(t[i]-predelay)/decay)
#    if(lpfilter > 0)
#        fsamp = 1000/samplerate; # get sampling frequency
#        fco = lpfilter;		# cutoff frequency in Hz
#        wco = fco/(fsamp/2); # wco of 1 is for half of the sample rate, so set it like this...
#        if(wco < 1) # if wco is > 1 then this is not a filter!
#            [b, a] = butter(8, wco);
#            template = filter(b, a, template); # filter each trace...
#        end;
#    end;
#
		m=np.amax(template)
		template = template/m # normalize to size 1.
		return(template, predelay)

	if funcid == 3: # dual exponential function with power
		for i in range(0,N):
			if t[i] >= predelay:
				td = t[i]-predelay
				template[i] = ((1.0-exp((-td/rise)))**2.0) * exp(-td/decay);
		m=np.amax(template)
		template = template/m # normalize to size 1.
		return(template, predelay)

	if funcid == 4: # use data from average waveform somewhere
		pass
	#    # 5/13/2008.
    #    # aveargin/zoom, etc put the displayed data into the "stack" variable
    #    # this is then usable as a template. It is normalized first, and
    #    # sent to the routine. 
    #    predelay = 0.05;
    #    template = STACK{1}; # pull from the new "stack" variable. 
    #    template = template/max(template);
	if funcid == 5: # use the average of the prior analysis.
		pass
		#    sf = getmainselection();
    #    predelay = CONTROL(sf).EPSC.predelay;
    #    template = CONTROL(sf).EPSC.iavg;
    #    template = template/max(template);
    #    
    #otherwise
    #    template = [];
    #    end;

def cb_multi(data, sign='+', samplerate=0.1, rise=[5.0, 3.0, 2.0, 0.5], decay=[30.0, 9.0, 5.0, 1.0],
			matchflag = True, threshold=3.0,
			dispflag=True, lpfilter = 0, template_type = 1, ntau=5):
	""" return the best choice of fits among a set of rise and fall times using the CB method.
		the tested rise and decay times are determined by the lists of rise and decay arrays.
		if matchflag is true, then the arrays are compared in order (rise[0], decay[0]),
		(rise[1], decay[1]). if matchflag is false, then all pairwise comparisons are made"""

	clist = [(1,0,0,1), (0,1,0,1), (0,0,1,1), (1,1,0,1), (1,0,1,1), (0,1,1,1),
			(1, 0.5, 0,5, 1), (0.5, 1, 0.5, 1), (0.5, 0.5, 1, 1), (0,0,0,1)]
	
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
	datan = data.copy() # datan will be modified during CB if subtractMode is on
	for k in range(0, nrTests): # use multiple template shapes
		for j in range(0, ndTests):
			if matchflag is True and j != k:
				continue
			(ic, pks, critval, isc, io, template) = ClementsBekkers(datan,
				samplerate = samplerate, rise = rise[k], decay = decay[j], threshold = threshold, sign = sign, 
				dispFlag = dispflag, subtractMode = True,
				lpfilter = lpfilter, template_type = template_type, ntau = ntau, markercolor=clist[k])
#	returns :: (eventlist, pkl, crit, scale, cx)
			if ic is  []:
				continue
			icand = np.append(icand, ic)
			peaks = np.append(peaks, pks)
			crit = np.append(crit, critval)
			iscamp = np.append(iscamp, isc)
			ioff = np.append(ioff, io)
			itj = np.append(itj, j*np.ones(len(ic)))
			itk = np.append(itk, k*np.ones(len(ic)))

	dist = 10.0 # minimum time bewteen events is set to 5 msec here.
	# pairwise comparision
	if sign is '-':
		ksign = -1
	else:
		ksign = 1
	print np.shape(icand)
	nt = len(icand)
	if nt is 0:
		return
	for ic in range(0,nt): # choose the best fit candidate events within dist of each other
		for jc in range(ic+1,nt): # compare each candidate template with the others
			if icand[jc] is -1 or icand[ic] is -1:
				continue
			if abs(icand[ic]- icand[jc]) < dist or abs(peaks[ic] - peaks[jc]) < dist:
				if ksign*crit[ic] > ksign*crit[jc]:
					icand[jc] = -1 # removes an event from the list
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
	pl.figure(2)
	t = samplerate*np.arange(0,len(data))
	pl.subplot(1,1,1);
	pl.plot(t, data, 'k', zorder = 0);
	pl.hold(True)
	ipts = icand.astype(int).tolist()
	ippts = peaks.astype(int).tolist()
	ijp = itj.astype(int).tolist()
	cols = []
	for p in range(0, len(ippts)):
		cols.append(clist[ijp[p]]) # plots below were t[ipts], data[ipts]
	pl.scatter(t[ipts], ioff, s=49, c=cols, marker='s', zorder=1);
	pl.scatter(t[ippts], iscamp, s=49, c=cols, marker = 'o', zorder=2);
	pl.show()

	return(icand, peaks, crit, iscamp, ioff)


if __name__ == "__main__":
	""" provide test of the selected routine(s) """
	(data, samplerate, rise, decay, threshold, sign) = testdata()
	ClementsBekkers(data, dispFlag = True, samplerate = samplerate,
					subtractMode = True, direction=-1,
					template_type = 1, rise=rise, decay=decay, threshold=4.0,
					ntau=8, sign=sign)
	pl.show()
##	cb_multi(data, sign=sign, matchflag = True,  dispflag = True, threshold = 4.5)