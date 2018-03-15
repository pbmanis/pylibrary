#!/usr/bin/env python
# encoding: utf-8
"""
RSTATS.py
Wrappers for using R, scipy and a custom permutation routine for data anlsysis

If you use the standard anaconda install for r, you will only get some packages.
To get permTS, do: conda install -c r r-perm

then importr('perm'), access with perm.permTS()

Currently implements 1-way ANOVA for 3, 4, and 5 groups with all-way posttests
t-tests (paired, unpaired, equal and unequal variance)
non-parametric tests
permutation

To use:
import pylibrary.RStats as RStats

Requires
Created by Paul Manis on 2014-06-26.

Copyright 2010-2014  Paul Manis 
Distributed under MIT/X11 license. See license.txt for more infofmation.

3-2018: Some changes made for Python portability: use list(dict.keys()) to get the keys as a list;
otherwise they are a "dict_keys" objects
Also, the ".next(iter)" iteratior was changed to __next__ in py3, but "next(iter)" works in boht.

"""
from __future__ import print_function
import scipy.stats as Stats
import numpy as np


# connect to R via RPy2
#try:
from rpy2.robjects import FloatVector
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

numpy2ri.activate()
RStats = importr('stats')
perm = importr("perm", "/Library/Frameworks/R.framework/Versions/3.4/Resources/library")  # may need to point to lib on system...

R_imported = True
#robjects.r.options(digits = 7)
#   RKMD = importr('kruskalmc')
print('R imported OK')
#except:
#    raise Exception ('RSTATS.py: R import Failed! Are R and RPy2 installed?')
#    R_imported = False
#    exit()

def pformat(p):
    """
    Take a value for p and format it differently depending on the value itself
    p = 1.0: '1.0'
    p = [0.1 -1.0]: '0.nn'
    p = [0.01 - 0.1]: '0.0nnn'
    p = [0.001 - 0.01]: '0.00nnn'
    p = [0.0001 - 0.001]: '0.00nnn'
    p < 1e-3: 'n.nnE-m'
    Parameters
    ----------
    p : the p value

    """
    fbrks = [1.0, 0.1, 0.01, 0.001, 0.0001]
    prec = [2, 3, 4, 5, 6]
    for i, f in enumerate(fbrks[:-1]):
        if fbrks[i] >= p and p > fbrks[i+1]:
            return '{:.{prec}f}'.format(p, prec=prec[i])
    return '{:.2e}'.format(p)


def KS(dataDict=None, dataLabel='data', mode='two.sided'):
    """
    KS performs a two-sample Kolmogorov-Smirnov test using the R package

    Params
    ------
    dataDict: Dictionary
        Data format {'group1': [dataset], 'group2': [dataset]}.
    dataLabel: string
        title to use for print out of data

    Returns
    -------
    (p, n) : tuple
        p value for test (against null hypothesis), and n, number of mc
         replications or 0 for other tests
    """

    # test calling values
    modes = ['two.sided', 'less', 'greater']
    if mode not in modes:
        raise ValueError('RSTATS.KS: mode must be in: ', modes)
    if (dataDict is None or not isinstance(dataDict, dict) 
            or len(dataDict.keys()) != 2):
        raise ValueError('RSTATS.KS: dataDict must be a dictionary with '
            + 'exactly 2 keys')

    labels = list(dataDict.keys())
#    NGroups = len(labels)
    cmdx = 'X=c(%s)' % ', '.join(str(x) for x in dataDict[labels[0]])
    cmdy = 'Y=c(%s)' % ', '.join(str(y) for y in dataDict[labels[1]])
# package "perm" not available on mac os x, use coin instead
#    importr('perm')
    robjects.r(cmdx)
    robjects.r(cmdy)

#    (pvalue, nmc) = permutation(dataDict, dataDict.keys())

    u = robjects.r("ks.test(X, Y, alternative='%s')" % mode)

    pvalue = float(u[1][0])
    statvalue = float(u[0][0]) # get diff estimate
    if dataLabel is not None:
        print ('\nKolmogorov-Smirnov test. Dataset = %s' % (dataLabel))
        print(u'  Test statistic: {:8.4f}'.format(statvalue))
        print(u'  p={:8.6f}, [mode={:s}]'.format(float(pvalue), mode))
    return pvalue


def OneWayAnova(dataStruct=None, dataLabel='data', mode='parametric'):
    """
    Perform a one way ANOVA with N groups

    Parameters
    ----------
    dataDict : Dictionary holding data (default: None)
        Format: {'group1': [dataset], 'group2': [dataset], 'group3': [dataset]}.
    dataLabel : string (default: 'data')
        title to use for print out of data
    mode : string (default: 'parametric')
        Choices: 'parametric' uses aov from R
                 'nonparametric' uses Kruskal-Wallis from R
    Prints resutls and post-hoc tests...

    Returns
    -------
    Nothing
    """
    # test calling values
    if mode not in ['parametric', 'nonparametric']:
        raise ValueError('RSTATS.OneWayAnova: Mode must be either parametric'+
                         ' or nonparametric; got %s' % mode)
    if dataStruct is None or not (isinstance(dataStruct, dict) or
             len(dataStruct.keys()) < 2):
        raise ValueError('RSTATS.OneWayAnova: dataStruct must be a dictionary'
            + ' with at least 2 keys')


    NGroups = len(dataStruct.keys())
    if NGroups <= 2:
        print("Need at least 3 groups to run One Way Anova")
        return
    data = [[]]*NGroups
    dn = [[]]*NGroups
    gn = [[]]*NGroups
    for i, d in enumerate(dataStruct.keys()):
        data[i] = dataStruct[d]

    print('OneWayAnova for %s ' % (dataLabel))
    print('datadict: ', len(dataStruct.keys()))
    labels = []
    print (labels)
    for i, d in enumerate(dataStruct.keys()): # range(NGroups):
        dn[i] = dataStruct[d] # dataStruct[dataStruct.keys()[i]]
        labels.append(d)
    dataargs = ', '.join([('dn[{:d}]'.format(d)) for d in range(NGroups)])
    (F, p) = eval('Stats.f_oneway(%s)' % dataargs)
    print('\nOne-way Anova (Scipy.stats), {0:d}'
        + ' Groups for check: F={1:f}, p = {2:8.4f}'.format(NGroups, F, p))
    if R_imported is False:
        'R not found, skipping'
        return
    print('\nR yields: ')
    for i, d in enumerate(dataStruct.keys()): # range(NGroups):
        gn[i] = FloatVector(dn[i])
        robjects.globalenv['g%d'%(i+1)] = gn[i]
        robjects.globalenv['d%d'%(i+1)] = dn[i]
        robjects.globalenv['l%d'%(i+1)] = d  # labels[i]
    cargs = ', '.join([('g{:d}'.format(d+1)) for d in range(NGroups)])
    clenargs = ', '.join([('length(g{:d})'.format(d+1)) for d in range(NGroups)])
    clargs = ', '.join([('l{:d}'.format(d+1)) for d in range(NGroups)])
    x = robjects.r("Yd <- c(%s)" % cargs) # could do inside call,
#             but here is convenient for testing values
    vData = robjects.r(("""vData <- data.frame( Yd,"""
                   + """Equation=factor(rep(c(%s), times=c(%s))))""" )% (clargs, clenargs))
# OLD code for blanaced groups:
    #groups = RBase.gl(3, len(d3), len(d3)*3, labels=labels)
    robjects.globalenv["vData"] = vData
    #weight = g1 + g2 + g3 # !!! Order is important here, get it right!
   # robjects.globalenv["weight"] = weight
   # robjects.globalenv["group"] = groups

    if mode == 'parametric':
        for i in range(len(data)):
            w, p = Stats.shapiro(data[i])
            if p > 0.05:
                print (('***** Data Set <{:s}> failed normality'
                    + ' (Shapiro-Wilk p = {:.2f}, w = {:.3f})')
                    .format(labels[i], p, w))
            else:
                print (('Data Set <{0:s}> passed normality'
                    + ' (Shapiro-Wilk p = {1:.2f}, w = {2:.3f})').
                    format(labels[i], p, w))
        aov = robjects.r("aov(Yd ~ Equation, data=vData)")
        print(aov)
        robjects.globalenv["aov"] = aov
        sum = robjects.r("summary(aov)")
        print(sum)
       # drop = robjects.r('drop1(aov, ~., test="F")')
        #print drop
        #x=RSTATS.anova(aov)
        #print x
        print('Post Hoc (Tukey HSD): ')
        y = robjects.r("TukeyHSD(aov)")
        print(y)
    if mode == 'nonparametric':
        kw = robjects.r("kruskal.test(Yd ~ Equation, data=vData)")
        print(kw)
#        loc = robjects.r('Equation=factor(rep(c(%s), times=c(%s)))' % (clargs, clenargs))
        kwmc = robjects.r('pairwise.wilcox.test'
                         + '(Yd, Equation, p.adj="bonferroni")')
        print ("PostTest: pairwise wilcox\n", kwmc)


def permTS(dataDict=None, dataLabel='data', mode='exact.ce'):
    """
    permTS performs a two-sample permutation test using the 'perm' package in R
    Uses the Monte-Carlo simulation method - not necessarily the fastest,
    but is "exact"
    This routine is a wrapper for permTS in R.

    Params
    ------
    dataDict: Dictionary
        Data format {'group1': [dataset], 'group2': [dataset]}.
    dataLabel: string
        title to use for print out of data
    mode: string
        test mode (see manual. Usually 'exact.ce' for "complete enumeration", or
        'exact.mc' for montecarlo)
    Returns
    -------
    (p, n) : tuple
        p value for test (against null hypothesis), and n, number of mc 
        replications or 0 for other tests
    """

    # test calling values
    if mode not in ['exact.ce', 'exact.mc']:
        raise ValueError('RStats.permTS: Mode must be either'
                        + ' "exact.ce" or "exact.mc"; got %s' % mode)
    if dataDict is None or not (isinstance(dataDict, dict) 
            or len(dataDict.keys()) != 2):
        raise ValueError('RSTATS.permTX: dataDict must be'
                        + ' a dictionary with exactly 2 keys')
    k = list(dataDict.keys())
    g1 = dataDict[k[0]]
    g2 = dataDict[k[1]]

    u = perm.permTS(
                FloatVector(g1), FloatVector(g2), alternative='two.sided',
                method=mode)
    pvalue = float(u[3][0])
    if mode == 'exact.mc':
        nmc = int(u[10][0])
    else:
        nmc = 0
    d = u[1].items()  # stored as a generator (interesting...)  # using next for py2/3
    estdiff = next(d)  #.next()  # gets the tuple with what was measured, and the value  
    if dataLabel is not None:
        print('\nPermutation Test (R permTS). Dataset = %s' % (dataLabel))
        print(u'  Test statistic: ({:s}): {:8.4f}'.
                format(estdiff[0], estdiff[1]))
        print(u'  p={:8.6f}, Nperm={:8d} [mode={:s}]'.
                format(float(pvalue), int(nmc), mode))
    return (pvalue, nmc)  # return the p value and the number of mc replicatess


def permutation(data, dataLabel=None, nperm=10000, decimals=4):
    """
    Brute force permutation test.

    Parameters
    ------
    dataDict : Dictionary
        Data format {'group1': [dataset], 'group2': [dataset]}.
    dataLabel : string
        title to use for print out of data
    nperm : int (default 10000)
        Number of permutation trials to run to calculate probabilities
    decimals : int (default 4)
        decimals in formatted printout
    Returns
    -------
    (p, n) : tuple
        p value for test (against null hypothesis), and n, 
        number permutations run
    """

    # test calling values
    if data is None or not isinstance(data, dict) or len(data.keys()) != 2:
        raise ValueError('RSTATS.permutation: data must be'
                + ' a dictionary with at exactly 2 keys'
                + '\nUse KW (anova) for more than 2 groups')

    k = list(data.keys())

    g1 = data[k[0]]
    g2 = data[k[1]]
    # (w1, p1) = Stats.shapiro(g1, a=None, reta=False)
    # (w2, p2) = Stats.shapiro(g2, a=None, reta=False)

    combined = np.concatenate((g1, g2))
    diffobs = np.mean(g2)-np.mean(g1)
    diffs = np.zeros(nperm)
    nperm = nperm
    index = range(0, combined.shape[0])
    for i in range(nperm):
        # draw from combined data set without replacement
        #shuff = np.random.randint(combined.shape[0], size=combined.shape[0])
        shuff = np.random.permutation(index)
        ar = combined[shuff[0:len(g1)]]
        br = combined[shuff[len(g1):]]
        diffs[i] = np.mean(br) - np.mean(ar)
    pvalue = np.sum(np.abs(diffs) >= np.abs(diffobs)) / float(nperm)
    if dataLabel is not None:
        print ('\n%s:  Permutation Test (Nperm = %d)' % (dataLabel, nperm))
        # if p1 < 0.05 and p2 < 0.05:
        #     print(u'  Both data sets appear normally distributed: Shapiro-Wilk Group 1 p = {:6.3f}, Group2 p = {:6.3f}'.format(p1, p2))
        # else:
        #     print(u'  ****At least one Data set is NOT normally distributed****\n      Shapiro-Wilk Group 1 p = {:6.3f}, Group2 p = {:6.3f}'.format(p1, p2))
        # print (u'    (Permutation test does not depend on distribution)')
    
        n = max([len(l) for l in k])
        print(u'  {:s}={:8.{pc}f} \u00B1{:.{pc}f}, {:d} (mean, SD, N)'.
                format(k[0].rjust(n), np.mean(g1), np.std(g1, ddof=1),
                       len(g1), pc=decimals))
        print(u'  {:s}={:8.{pc}f} \u00B1{:.{pc}f}, {:d} (mean, SD, N)'.
                format(k[1].rjust(n), np.mean(g2), np.std(g2, ddof=1),
                        len(g2), pc=decimals))
        summarizeData(data, decimals=decimals)
        # iqr1 = np.subtract(*np.percentile(g1, [75, 25]))
        # iqr2 = np.subtract(*np.percentile(g2, [75, 25]))
        # print(u'  {:s}: median={:8.4f}  IQR={:8.4f}'.format(k[0].rjust(n), np.median(g1), iqr1))
        # print(u'  {:s}: median={:8.4f}  IQR={:8.4f}'.format(k[1].rjust(n), np.median(g2), iqr2))
        print(u'  Observed difference: {:8.4f}'.format(diffobs))
        print(u'  p={:8.6f}, Nperm={:8d}\n'.format(float(pvalue), int(nperm)))
    return(pvalue, nperm)


def ttest(
        data, dataLabel=None, paired=False, decimals=4,
        textline=False, units=None
        ):
    """
    Perform a t-test using Scipy.stats

    Comment: This routine first tests the equal varaince assumption.
    There are reasons that this might be viewed askance.
    Therefore, a simple test just using Welch's test, and
        assuming unequal variance may be prefered (Ruxton, G. Behav. Ecol.,
        17:688, 2006)
    The current version no longer performs a prior test for equal variances,
        nor does it report the results for that test.
    We always assume variances are unequal.
    Parameters
    ----------
    data : Dictionary (default: None)
        Data format {'group1': [dataset], 'group2': [dataset]}.
    dataLabel : string (default: None)
        title to use for print out of data
    paired : Boolean (default: False)
        Set true to do "paired" t-test, false to do unpaired 
        independent samples) test.
    decimals : int (default 4)
        decimals in formatted printout

    Returns
    -------
     (df, t, p) : tuple
        df : degrees of freedom calculated assuming unequal variance
        t : t statistic for the difference
        p : p value
    """

    # test calling values
    if data is None or not isinstance(data, dict) or len(data.keys()) != 2:
        raise ValueError('RSTATS.ttest: data must be a dictionary'
                + ' with at exactly 2 keys'
                + '\nUse KW (anova) for more than 2 groups')

    k = list(data.keys())
    g = {}
    n = {}
    gmean = {}
    gstd = {}

    g[1] = data[k[0]]
    g[2] = data[k[1]]
    n[1] = len(g[1])
    n[2] = len(g[2])
    # (w1, p1) = Stats.shapiro(g1, a=None, reta=False)
    # (w2, p2) = Stats.shapiro(g2, a=None, reta=False)
    # Tb, pb = Stats.bartlett(g1, g2)  # do bartletss for equal variance
    equalVar = False

    if paired:
        print (len(g[1]), len(g[2]))
        (t, p) = Stats.ttest_rel(g[1], g[2])
    else:
        (t, p) = Stats.ttest_ind(g[1], g[2], equal_var=equalVar)
    gmean[1] = np.mean(g[1])
    gstd[1] = np.std(g[1], ddof=1)
    gmean[2] = np.mean(g[2])
    gstd[2] = np.std(g[2], ddof=1)
    #       df = (tstd[k]**2/tN[k] + dstd[k]**2/dN[k])**2 / (( (tstd[k]**2 /
    # tN[k])**2 / (tN[k] - 1) ) + ( (dstd[k]**2 / dN[k])**2 / (tN[k] - 1) ) )
    df = ((gstd[1]**2/n[1] + gstd[2]**2/n[2])**2
            / (((gstd[1]**2 / n[1])**2 / (n[1] - 1)
            + ((gstd[2]**2 / n[2])**2 / (n[1] - 1))))
            )
    if dataLabel is not None:
        testtype = 'Independent'
        if paired:
            testtype = 'Paired'
        n = max([len(l) for l in k])
        print ('\n%s\n  %s T-test, Welch correction' % (dataLabel, testtype))
        # if p1 < 0.05 and p2 < 0.05:
        #     print(u'  Both data sets appear normally distributed: Shapiro-Wilk Group 1 p = {:6.3f}, Group2 p = {:6.3f}'.format(p1, p2))
        # else:
        #     print(u'  ****At least one Data set is NOT normally distributed****\n      Shapiro-Wilk Group 1 p = {:6.3f}, Group2 p = {:6.3f}'.format(p1, p2))
        #     print (u'    (performing test anyway, as requested)')
        # if equalVar:
        #     print(u'  Variances are equivalent (Bartletts test, p = {:.3f})'.format(pb))
        # else:
        #     print(u'  Variances are unequal (Bartletts test, p = {:.3f}); not assuming equal variances'.format(pb))
        print(u'  {:s}={:8.{pc}f} (SD {:.{pc}f}, N = {:d})'.
                format(k[0].rjust(n), gmean[1], gstd[1],
                len(g[1]), pc=decimals))
        print(u'  {:s}={:8.{pc}f} (SD {:.{pc}f}, N = {:d})'.
                format(k[1].rjust(n), gmean[2], gstd[2],
                len(g[2]), pc=decimals))
        print(u'  t({:6.2f})={:8.4f}  p={:8.6f}\n'.
                format(df, float(t), float(p)))
        # generate one line of text suitable for pasting into a paper
        if textline:
            if units is not None:
                units = ' ' + units
            else:
                units = ''
            fmtstring = u'{:s}: {:.{pc}f} (SD {:.{pc}f}, N={:d}){:s}; '
            print(u'(', end='')
            for s in range(1, 3):
                print(fmtstring.format(
                    k[s-1], gmean[s], gstd[s], len(g[s]), units, 
                    pc=decimals), end='')
            print(u't{:.2f}={:.3f}, p={:s})\n'.format(df, float(t), pformat(p)))

    return(df, float(t), float(p))


def ranksums(data, dataLabel=None, paired=False, decimals=4):
    """
    Perform a rank sums test using Scipy.stats
    Use for non-parametric data sets

    Parameters
    ----------
    data : Dictionary (default: None)
        Data format {'group1': [dataset], 'group2': [dataset]}.
    dataLabel : string (default: None)
        title to use for print out of data
    paired : Boolean (default: False)
        Set true to do "paired" t-test, false to do unpaired (independent
        samples) test.
    decimals : int (default 4)
        decimals in formatted printout

    Returns
    -------
     (df, t, p) : tuple
        df : degrees of freedom calculated assuming unequal variance
        t : t statistic for the difference
        p : p value
    """
    if data is None or not isinstance(data, dict) or len(data.keys()) != 2:
        raise ValueError('RSTATS.permutation: data must be a dictionary with'
            + ' at exactly 2 keys' +
            '\nUse KW (anova) for more than 2 groups')

    k = list(data.keys())
#    labels = data.keys()
    g1 = data[k[0]]
    g2 = data[k[1]]
    # n1 = len(g1)
    # n2 = len(g2)

    if paired:
        (z, p) = Stats.wilcoxon(g1, g2)
        res = RStats.wilcox_test(g1, g2, pair=True)
        testtype = "Wilcoxon signed-rank"
        pairedtype = "Paired"
    else:
        (z, p) = Stats.ranksums(g1, g2)
        testtype = "Rank-sums test"
        res = RStats.wilcox_test(g1, g2, pair=False)
        pairedtype = "Independent"

    g1mean = np.mean(g1)
    g1std = np.std(g1, ddof=1)
    g2mean = np.mean(g2)
    g2std = np.std(g2, ddof=1)
    (w1, p1) = Stats.shapiro(g1) #, a=None, reta=False)
    (w2, p2) = Stats.shapiro(g2) #, a=None, reta=False)
    if dataLabel is not None:
        n = max([len(l) for l in k])
        print('\n%s test, data set = %s' % (testtype, dataLabel))
        if p1 < 0.05 and p2 < 0.05:
            print(u'  Both data sets appear normally distributed: \n'
                + '    Shapiro-Wilk Group 1 p = {:6.3f}, Group2 p = {:6.3f}'.
                format(p1, p2))
        else:
            print(u'  ***At least one Data set is NOT normally distributed***\n'
                + '    Shapiro-Wilk Group 1 p = {:6.3f}, Group2 p = {:6.3f}'.
                format(p1, p2))
            print(u'    (RankSums does not assume normal distribution)')
    
        print(u'  {:s}={:8.{pc}f}\u00B1{:.{pc}f}, {:d} (mean, SD, N)'.
                format(k[0].rjust(n), g1mean, g1std, len(g1), pc=decimals))
        print(u'  {:s}={:8.{pc}f}\u00B1{:.{pc}f}, {:d} (mean, SD, N)'.
                format(k[1].rjust(n), g2mean, g2std, len(g2), pc=decimals))
        summarizeData(data, decimals=decimals)
        # iqr1 = np.subtract(*np.percentile(g1, [75, 25]))
        # iqr2 = np.subtract(*np.percentile(g2, [75, 25]))
        # print(u'  {:s}: median={:8.4f}  IQR={:8.4f}'.
        #       format(k[0].rjust(n), np.median(g1), iqr1))
        # print(u'  {:s}: median={:8.4f}  IQR={:8.4f}'.
        #        format(k[1].rjust(n), np.median(g2), iqr2))
        print(u'  z={:8.4f}   p={:8.6f}   <scipy.Stats: {:20s}, {:11s}>'.
                format(float(z), float(p), testtype, pairedtype))
        print(u'  z={:8.4f}   p={:8.6f}   <R Stats    : {:20s}, {:11s}>\n'.
                format(res[res.names.index('statistic')][0],
                float(res[res.names.index('p.value')][0]),
                testtype, pairedtype))
    return(float(z), float(p))


def summarizeData(data, dataLabel=None, decimals=4):
    """
    Provide descriptive median and interquartile range for non-normal
    (or small) data sets
    """
    if dataLabel is not None:
        print ('%s: Data Set Summary (median, IQR)' % dataLabel)
    n = max([len(l) for l in data.keys()])
    for i, k in enumerate(data.keys()):
        g1 = data[k]
        iqr1 = np.subtract(*np.percentile(g1, [75, 25]))
        print(u'  {:s}:  {:8.{pc}f}, {:.{pc}f} (median, IQR)'.
                format(k.rjust(n), np.median(g1), iqr1, pc=decimals))


def testANOVA():
    """
    Tests for anova routines
    3 group anova taken from Prism 6.01 example
    Prints results for direct comparisoin with the Prism output, which
    is copied below.

    Parameters
    ----------
    None

    Returns
    -------
    Nothing
    """

    data = {'Control': [54, 23, 45, 54, 45, 47], 'Treated': [87, 98, 64, 77, 89],
    'TreatedAntagonist': [45, 39, 51, 49, 50, 55]}
    print(type(data))
    OneWayAnova(dataStruct=data, dataLabel='3 Groups', mode='parametric')
    print ('-'*80)
    print ('Compare to Prism output: ')
    print( """
    "Table Analyzed"	"One-way ANOVA data"

    "ANOVA summary"
    "  F"	22.57
    "  P value"	"< 0.0001"
    "  P value summary"	****
    "  Are differences among means statistically significant? (P < 0.05)"	Yes
    "  R square"	0.7633

    "Brown-Forsythe test"
    "  F (DFn, DFd)"	"0.7307 (2, 14)"
    "  P value"	0.4991
    "  P value summary"	ns
    "  Significantly different standard deviations? (P < 0.05)"	No

    "Bartlett's test"
    "  Bartlett's statistic (corrected)"	2.986
    "  P value"	0.2247
    "  P value summary"	ns
    "  Significantly different standard deviations? (P < 0.05)"	No

    "ANOVA table"	SS	DF	MS	"F (DFn, DFd)"	"P value"
    "  Treatment (between columns)"	4760	2	2380	"F (2, 14) = 22.57"	"P < 0.0001"
    "  Residual (within columns)"	1476	14	105.4
    "  Total"	6236	16

    "Data summary"
    "  Number of treatments (columns)"	3
    "  Number of values (total)"	17				""")
    print ('-'*80)
    print ('Multiple comparisions from Prism:')
    print ("""
    "Number of families"	1
    "Number of comparisons per family"	3
    Alpha	0.05

    "Tukey's multiple comparisons test"	"Mean Diff."	"95% CI of diff."	Significant?	Summary

    "  Control vs. Treated"	-38.33	"-54.61 to -22.06"	Yes	****		A-B
    "  Control vs. Treated+Antagonist"	-3.500	"-19.02 to 12.02"	No	ns		A-C
    "  Treated vs. Treated+Antagonist"	34.83	"18.56 to 51.11"	Yes	***		B-C


    "Test details"	"Mean 1"	"Mean 2"	"Mean Diff."	"SE of diff."	n1	n2	q	DF

    "  Control vs. Treated"	44.67	83.00	-38.33	6.218	6	5	8.719	14
    "  Control vs. Treated+Antagonist"	44.67	48.17	-3.500	5.928	6	6	0.8349	14
    "  Treated vs. Treated+Antagonist"	83.00	48.17	34.83	6.218	5	6	7.923	14

    """)


def test2Samp():
    """
    Perform tests on ranksums, permutation permTS and t-test, all on
    2 samples. Runs using a fixed random number seed (0) each time
    Prints results for each test.

    Parameters
    ----------
    None

    Returns
    -------
    Nothing.
    """

    sigmax = 1.0
    sigmay = 3.0
    mux = 0.0
    muy = 3.0
    nx = 10
    ny = 10
    # Update
    np.random.RandomState(0)  # set seed to 0
    datax = sigmax * np.random.randn(nx) + mux
    datay = sigmay * np.random.randn(ny) + muy
    datadict = {'x': datax, 'y': datay}
    ranksums(datadict, dataLabel='Test Rank Sums (scipy)')
    ranksums(datadict, dataLabel='Test Rank Sums, Paired (scipy)', paired=True)
    ttest(datadict, dataLabel='Standard t-test (scipy)', 
                    textline=True, decimals=3, units='mV')
    ttest(datadict, dataLabel='Standard t-test (scipy), paired', paired=True,
                    textline=True, decimals=3)
    (p, n) = permTS(datadict, dataLabel='R permTS')
    permutation(datadict, dataLabel='Test simple permute')
    KS(datadict, dataLabel='Test with KS')


if __name__ == '__main__':
    """
    If called from commmand line, runs tests on routines
    """
    for p in [0.25, 0.038, 0.0048, 0.00001, 1e-18]:
        print(pformat(p))

    testANOVA()
    test2Samp()
    