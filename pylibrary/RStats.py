#!/usr/bin/env python
# encoding: utf-8
"""
RStats.py
Wrappers for using R for data anlsysis
Currently implements 1-way ANOVA for 3, 4, and 5 groups with all-way posttests
Created by Paul Manis on 2014-06-26.

Copyright 2010-2014  Paul Manis
Distributed under MIT/X11 license. See license.txt for more infofmation.
"""
import scipy.stats
import numpy as np

# connect to R via RPy2
try:
    from rpy2.robjects import FloatVector
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    RStats = importr('stats')
    R_imported = True
    robjects.r.options("digite = 7")
#   RKMD = importr('kruskalmc')
    print 'R imported OK'
except:
    raise Exception ('Rstats.py: R import Failed! Are R and RPy2 installed?')
    R_imported = False
    exit()


def OneWayAnova(dataDict=None, dataLabel='data', mode='parametric'):
    """
    Perform a one way ANOVA with N groups
    :param dataDict: Dictionary holding data in format
        {'group1': [dataset], 'group2': [dataset], 'group3': [dataset]}.
    :param dataLabel: title to use for print out of data
    :param mode: 'parametric' (default) uses R aov. anything else uses Kruskal-Wallis
    Prints resutls and post-hoc tests...
    :return: Nothing
    """ 
    labels = dataDict.keys()
    NGroups = len(labels)
    if NGroups <= 2:
        print "Need at least 3 groups to run One Way Anova"
        return
    data = [[]]*NGroups
    dn = [[]]*NGroups
    gn = [[]]*NGroups
    for i, d in enumerate(labels):
        data[i] = dataDict[d]

    print 'OneWayAnova for %s' % (dataLabel)
    for i in range(NGroups):
        dn[i]=dataDict[labels[i]]
    dataargs = ', '.join([('dn[{:d}]'.format(d)) for d in range(NGroups)])
    (F, p) = eval('scipy.stats.f_oneway(%s)' % dataargs)
    print '\nOne-way Anova (Scipy.stats), %d Groups for check: F=%f, p = %8.4f'% (NGroups, F, p)
    if R_imported is False:
        'R not found, skipping'
        return
    print '\nR yields: ' 
    for i in range(NGroups):
        gn[i] = FloatVector(dn[i])
        robjects.globalenv['g%d'%(i+1)] = gn[i]
        robjects.globalenv['d%d'%(i+1)] = dn[i]
        robjects.globalenv['l%d'%(i+1)] = labels[i]
    cargs = ', '.join([('g{:d}'.format(d+1)) for d in range(NGroups)])
    clenargs = ', '.join([('length(g{:d})'.format(d+1)) for d in range(NGroups)])
    clargs = ', '.join([('l{:d}'.format(d+1)) for d in range(NGroups)])
    x=robjects.r("Yd <- c(%s)" % cargs) # could do inside call, but here is convenient for testing values
    vData = robjects.r("""vData <- data.frame( Yd, 
                Equation=factor(rep(c(%s), times=c(%s))))""" % (clargs, clenargs))
# OLD code for blanaced groups:
    #groups = RBase.gl(3, len(d3), len(d3)*3, labels=labels)
    #robjects.globalenv["vData"] = vData
    #weight = g1 + g2 + g3 # !!! Order is important here, get it right!
   # robjects.globalenv["weight"] = weight
   # robjects.globalenv["group"] = groups

    if mode == 'parametric':
        for i in range(len(data)):
            w, p = scipy.stats.shapiro(data[i])
            if p > 0.05:
                print ('***** Data Set <{:s}> failed normality (Shapiro-Wilk p = {:.2f}, w = {:.3f})'.format(labels[i]))
            else:
                print ('Data Set <{:s}> passed normality (Shapiro-Wilk p = {:.2f}, w = {:.3f})'.format(labels[i]))
        aov = robjects.r("aov(Yd ~ Equation, data=vData)")
        print aov
        robjects.globalenv["aov"] = aov
        sum = robjects.r("summary(aov)")
        print sum
       # drop = robjects.r('drop1(aov, ~., test="F")')
        #print drop
        #x=RStats.anova(aov)
        #print x
        print 'Post Hoc (Tukey HSD): '
        y=robjects.r("TukeyHSD(aov)")
        print y
    else:
        kw = robjects.r("kruskal.test(Yd ~ Equation, data=vData)")
        print kw
        loc = robjects.r('Equation=factor(rep(c(%s), times=c(%s)))' % (clargs, clenargs))
        kwmc = robjects.r('pairwise.wilcox.test(Yd, Equation, p.adj="bonferroni")')
        print "PostTest: pairwise wilcox\n", kwmc
    #rprint = robjects.globalenv.get("print")
    # the boxplot is redundant, but worth checking out
    #bp = robjects.r("boxplot(Yd ~ Group, data=vData)")
    #rprint(bp)
#        import time
#        time.sleep(10)
#
#
#
# def OneWayAnova3(dataDict=None, dataLabel='data', mode='parametric'):
#     """
#     Perform a one way ANOVA with 3 groups
#     :param dataDict: Dictionary holding data in format
#         {'group1': [dataset], 'group2': [dataset], 'group3': [dataset]}
#     :param dataLabel: title to use for print out of data
#     :param mode: 'parametric' (default) uses R aov. anything else uses Kruskal-Wallis
#     Prints resutls and post-hoc tests...
#     :return: Nothing
#     """
#     labels = dataDict.keys()
#     data = [[]]*len(labels)
#     for i, d in enumerate(labels):
#         data[i] = dataDict[d]
#     if len(data) < 3:
#         print "Need 3 groups to run One Way Anova here"
#         return
#     print 'OneWayAnova for %s' % (dataLabel)
#     da1=dataDict[labels[0]]
#     da2=dataDict[labels[1]]
#     da3=dataDict[labels[2]]
#     (F, p) = scipy.stats.f_oneway(da1, da2, da3)
#     print '\nOne-way Anova (Scipy.stats): F=%f, p = %8.4f'% (F, p)
#     if R_imported is False:
#         print 'R not found, skipping'
#         return
#     print '\nR yields (anova3): '
#     g1 = FloatVector(da1)
#     g2 = FloatVector(da2)
#     g3 = FloatVector(da3)
#     robjects.globalenv['g1'] = g1
#     robjects.globalenv['g2'] = g2
#     robjects.globalenv['g3'] = g3
#     #print da1
#     #print dir(da1)
#     robjects.globalenv['d1'] = FloatVector(da1)
#     robjects.globalenv['d2'] = FloatVector(da2)
#     robjects.globalenv['d3'] = FloatVector(da3)
#     robjects.globalenv['d3'] = da3
#     robjects.globalenv['l1'] = labels[0]
#     robjects.globalenv['l2'] = labels[1]
#     robjects.globalenv['l3'] = labels[2]
#     x=robjects.r("Yd <- c(g1, g2, g3)") # could do inside call, but here is convenient for testing values
#     vData = robjects.r("""vData <- data.frame( Yd,
#                 Group=factor(rep(c(l1, l2, l3), times=c(length(g1), length(g2), length(g3)))))""")
# # OLD code for blanaced groups:
#     #groups = RBase.gl(3, len(d3), len(d3)*3, labels=labels)
#     #robjects.globalenv["vData"] = vData
#     #weight = g1 + g2 + g3 # !!! Order is important here, get it right!
#    # robjects.globalenv["weight"] = weight
#    # robjects.globalenv["group"] = groups
#
#     if mode == 'parametric':
#         for i in range(len(data)):
#             w, p = scipy.stats.shapiro(data[i])
#             if p > 0.05:
#                 print ('***** Data Set <{:s}> failed normality (Shapiro-Wilk p = {:.2f}, w = {:.3f})'.format(labels[i]))
#             else:
#                 print ('Data Set <{:s}> passed normality (Shapiro-Wilk p = {:.2f}, w = {:.3f})'.format(labels[i]))
#         aov = robjects.r("aov(Yd ~ Group, data=vData)")
#         print aov
#         robjects.globalenv["aov"] = aov
#         sum = robjects.r("summary(aov)")
#         print sum
#         drop = robjects.r('drop1(aov, ~., test="F")')
#         print drop
#         #x=RStats.anova(aov)
#         #print x
#         print 'Post Hoc (Tukey HSD): '
#         y=robjects.r("TukeyHSD(aov)")
#         print y
#     else:
#         kw = robjects.r("kruskal.test(Yd ~ Group, data=vData)")
#         print kw
#         loc = robjects.r('Group=factor(rep(c(l1, l2, l3), times=c(length(g1), length(g2), length(g3))))')
#         kwmc = robjects.r('pairwise.wilcox.test(Yd, Group, p.adj="bonferroni")')
#         print "PostTest: pairwise wilcox\n", kwmc
#     #rprint = robjects.globalenv.get("print")
#     # the boxplot is redundant, but worth checking out
#     #bp = robjects.r("boxplot(Yd ~ Group, data=vData)")
#     #rprint(bp)
# #        import time
# #        time.sleep(10)
#
#
# def OneWayAnova4(dataDict=None, dataLabel='data', mode='parametric'):
#     """
#     Perform a one way ANOVA with 4 groups
#     :param dataDict: Dictionary holding data in format
#         {'group1': [dataset], 'group2': [dataset], 'group3': [dataset]}
#     :param dataLabel: title to use for print out of data
#     :param mode: 'parametric' (default) uses R aov. anything else uses Kruskal-Wallis
#     Prints resutls and post-hoc tests...
#     :return: Nothing
#     """
#     labels = dataDict.keys()
#     data = [[]]*len(labels)
#     for i, d in enumerate(labels):
#         data[i] = dataDict[d]
#     if len(data) < 4:
#         print "Need 4 groups to run One Way Anova 4 here"
#         return
#     print 'OneWayAnova for %s' % (dataLabel)
#     d1=dataDict[labels[0]]
#     d2=dataDict[labels[1]]
#     d3=dataDict[labels[2]]
#     d4=dataDict[labels[3]]
#     (F, p) = scipy.stats.f_oneway(d1, d2, d3, d4)
#     print '\nOne-way Anova (Scipy.stats), for check: F=%f, p = %8.4f'% (F, p)
#     if R_imported is False:
#         'R not found, skipping'
#         return
#     print '\nR yields: '
#     g1 = FloatVector(d1)
#     g2 = FloatVector(d2)
#     g3 = FloatVector(d3)
#     g4 = FloatVector(d4)
#     robjects.globalenv['g1'] = g1
#     robjects.globalenv['g2'] = g2
#     robjects.globalenv['g3'] = g3
#     robjects.globalenv['g4'] = g4
#     robjects.globalenv['d1'] = d1
#     robjects.globalenv['d2'] = d2
#     robjects.globalenv['d3'] = d3
#     robjects.globalenv['d4'] = d4
#     robjects.globalenv['l1'] = labels[0]
#     robjects.globalenv['l2'] = labels[1]
#     robjects.globalenv['l3'] = labels[2]
#     robjects.globalenv['l4'] = labels[3]
#     x=robjects.r("Yd <- c(g1, g2, g3, g4)") # could do inside call, but here is convenient for testing values
#     vData = robjects.r("""vData <- data.frame( Yd,
#                 Equation=factor(rep(c(l1, l2, l3, l4), times=c(length(g1), length(g2), length(g3), length(g4)))))""")
# # OLD code for blanaced groups:
#     #groups = RBase.gl(3, len(d3), len(d3)*3, labels=labels)
#     #robjects.globalenv["vData"] = vData
#     #weight = g1 + g2 + g3 # !!! Order is important here, get it right!
#    # robjects.globalenv["weight"] = weight
#    # robjects.globalenv["group"] = groups
#
#     if mode == 'parametric':
#         for i in range(len(data)):
#             w, p = scipy.stats.shapiro(data[i])
#             if p > 0.05:
#                 print ('***** Data Set <{:s}> failed normality (Shapiro-Wilk p = {:.2f}, w = {:.3f})'.format(labels[i]))
#             else:
#                 print ('Data Set <{:s}> passed normality (Shapiro-Wilk p = {:.2f}, w = {:.3f})'.format(labels[i]))
#         aov = robjects.r("aov(Yd ~ Equation, data=vData)")
#         print aov
#         robjects.globalenv["aov"] = aov
#         sum = robjects.r("summary(aov)")
#         print sum
#        # drop = robjects.r('drop1(aov, ~., test="F")')
#         #print drop
#         #x=RStats.anova(aov)
#         #print x
#         print 'Post Hoc (Tukey HSD): '
#         y=robjects.r("TukeyHSD(aov)")
#         print y
#     else:
#         kw = robjects.r("kruskal.test(Yd ~ Equation, data=vData)")
#         print kw
#         loc = robjects.r('Equation=factor(rep(c(l1, l2, l3, l4), times=c(length(g1), length(g2), length(g3), length(g4))))')
#         kwmc = robjects.r('pairwise.wilcox.test(Yd, Equation, p.adj="bonferroni")')
#         print "PostTest: pairwise wilcox\n", kwmc
#     #rprint = robjects.globalenv.get("print")
#     # the boxplot is redundant, but worth checking out
#     #bp = robjects.r("boxplot(Yd ~ Group, data=vData)")
#     #rprint(bp)
# #        import time
# #        time.sleep(10)
#
# def OneWayAnova5(dataDict=None, dataLabel='data', mode='parametric'):
#     """
#     Perform a one way ANOVA with 5 groups
#     :param dataDict: Dictionary holding data in format
#         {'group1': [dataset], 'group2': [dataset], 'group3': [dataset]}
#     :param dataLabel: title to use for print out of data
#     :param mode: 'parametric' (default) uses R aov. anything else uses Kruskal-Wallis
#     Prints resutls and post-hoc tests...
#     :return: Nothing
#     """
#     labels = dataDict.keys()
#     data = [[]]*len(labels)
#     for i, d in enumerate(labels):
#         data[i] = dataDict[d]
#     if len(data) < 4:
#         print "Need 4 groups to run One Way Anova 4 here"
#         return
#     print 'OneWayAnova for %s' % (dataLabel)
#     d1=dataDict[labels[0]]
#     d2=dataDict[labels[1]]
#     d3=dataDict[labels[2]]
#     d4=dataDict[labels[3]]
#     d5=dataDict[labels[4]]
#     (F, p) = scipy.stats.f_oneway(d1, d2, d3, d4, d5)
#     print '\nOne-way Anova (Scipy.stats), for check: F=%f, p = %8.4f'% (F, p)
#     if R_imported is False:
#         'R not found, skipping'
#         return
#     print '\nR yields: '
#     g1 = FloatVector(d1)
#     g2 = FloatVector(d2)
#     g3 = FloatVector(d3)
#     g4 = FloatVector(d4)
#     g5 = FloatVector(d5)
#     robjects.globalenv['g1'] = g1
#     robjects.globalenv['g2'] = g2
#     robjects.globalenv['g3'] = g3
#     robjects.globalenv['g4'] = g4
#     robjects.globalenv['g5'] = g5
#     robjects.globalenv['d1'] = d1
#     robjects.globalenv['d2'] = d2
#     robjects.globalenv['d3'] = d3
#     robjects.globalenv['d4'] = d4
#     robjects.globalenv['d5'] = d5
#     robjects.globalenv['l1'] = labels[0]
#     robjects.globalenv['l2'] = labels[1]
#     robjects.globalenv['l3'] = labels[2]
#     robjects.globalenv['l4'] = labels[3]
#     robjects.globalenv['l5'] = labels[4]
#     x=robjects.r("Yd <- c(g1, g2, g3, g4, g5)") # could do inside call, but here is convenient for testing values
#     vData = robjects.r("""vData <- data.frame( Yd,
#                 Equation=factor(rep(c(l1, l2, l3, l4, l5), times=c(length(g1), length(g2), length(g3), length(g4), length(g5)))))""")
# # OLD code for blanaced groups:
#     #groups = RBase.gl(3, len(d3), len(d3)*3, labels=labels)
#     #robjects.globalenv["vData"] = vData
#     #weight = g1 + g2 + g3 # !!! Order is important here, get it right!
#    # robjects.globalenv["weight"] = weight
#    # robjects.globalenv["group"] = groups
#
#     if mode == 'parametric':
#         for i in range(len(data)):
#             w, p = scipy.stats.shapiro(data[i])
#             if p > 0.05:
#                 print ('***** Data Set <{:s}> failed normality (Shapiro-Wilk p = {:.2f}, w = {:.3f})'.format(labels[i]))
#             else:
#                 print ('Data Set <{:s}> passed normality (Shapiro-Wilk p = {:.2f}, w = {:.3f})'.format(labels[i]))
#
#
#         aov = robjects.r("aov(Yd ~ Equation, data=vData)")
#         print aov
#         robjects.globalenv["aov"] = aov
#         sum = robjects.r("summary(aov)")
#         print sum
#        # drop = robjects.r('drop1(aov, ~., test="F")')
#         #print drop
#         #x=RStats.anova(aov)
#         #print x
#         print 'Post Hoc (Tukey HSD): '
#         y=robjects.r("TukeyHSD(aov)")
#         print y
#     else:
#         kw = robjects.r("kruskal.test(Yd ~ Equation, data=vData)")
#         print kw
#         loc = robjects.r('Equation=factor(rep(c(l1, l2, l3, l4, l5), times=c(length(g1), length(g2), length(g3), length(g4), length(g5))))')
#         kwmc = robjects.r('pairwise.wilcox.test(Yd, Equation, p.adj="bonferroni")')
#         print "PostTest: pairwise wilcox\n", kwmc
#     #rprint = robjects.globalenv.get("print")
#     # the boxplot is redundant, but worth checking out
#     #bp = robjects.r("boxplot(Yd ~ Group, data=vData)")
#     #rprint(bp)
# #        import time
# #        time.sleep(10)


def permutation(data, dataLabel=None, nperm=10000, decimals=4):
    k = data.keys()
    if len(k) > 2:
        raise ValueError('Permutation can only compare 2 groups: use KW (anova) for more than 2 groups')
        
    g1 = data[k[0]]
    g2 = data[k[1]]
    (w1, p1) = Stats.shapiro(g1, a=None, reta=False)
    (w2, p2) = Stats.shapiro(g2, a=None, reta=False)
    if p1 > 0.05 or p2 > 0.05:
        print(u'Some data are NOT normally distributed: Shapiro-Wilk Group 1 p = {:6.3f}, Group2 p = {:6.3f}'.format(p1, p2))
        print (u'    (Permutation test does not depend on distribution)')
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
        print 'Permutation Test. Dataset = %s (N = %d)' % (dataLabel, nperm)
        n = max([len(l) for l in k])
        print(u'  {:s}={:8.{pc}f} \u00B1{:.{pc}f} (mean, SD)'.format(k[0].rjust(n), np.mean(g1), np.std(g1), pc=decimals))
        print(u'  {:s}={:8.{pc}f} \u00B1{:.{pc}f} (mean, SD)'.format(k[1].rjust(n), np.mean(g2), np.std(g2), pc=decimals))
        print(u'   p={:8.6f}, N={:8d}'.format(float(pvalue), int(nperm)))
        summarizeData(data, decimals=decimals)
        # iqr1 = np.subtract(*np.percentile(g1, [75, 25]))
        # iqr2 = np.subtract(*np.percentile(g2, [75, 25]))
        # print(u'  {:s}: median={:8.4f}  IQR={:8.4f}'.format(k[0].rjust(n), np.median(g1), iqr1))
        # print(u'  {:s}: median={:8.4f}  IQR={:8.4f}'.format(k[1].rjust(n), np.median(g2), iqr2))
        print(u'Observed difference: {:8.4f}'.format(diffobs))
    return(pvalue, nperm)

def ttest(data, dataLabel=None, decimals=4):
    """
    perform a t-test using Scipy.stats
    :param data: dictionary {'g1': [data], 'g2': [data]}
    :param dataLabel: string with label to be printed for summary information
    :return df: degrees of freedom calculated assuming unequal variance
    :return t: t statistic for the difference
    :return p: p value
    """
    k = data.keys()
    g1 = data[k[0]]
    g2 = data[k[1]]
    n1 = len(g1)
    n2 = len(g2)
    (w1, p1) = Stats.shapiro(g1, a=None, reta=False)
    (w2, p2) = Stats.shapiro(g2, a=None, reta=False)
    if p1 < 0.05 and p2 < 0.05:
        print(u'Data appear normally distributed: Shapiro-Wilk Group 1 p = {:6.3f}, Group2 p = {:6.3f}'.format(p1, p2))
    else:
        print(u'****Data are NOT normally distributed: Shapiro-Wilk Group 1 p = {:6.3f}, Group2 p = {:6.3f} *****'.format(p1, p2))
        print (u'(performing test anyway, as requested)')
    Tb, pb = Stats.bartlett(g1, g2)  # do bartletss for equal variance
    if pb < 0.05:
        equalvar = True
        print(u'  Variances are equivalent (Bartletts test, p = %.3f)'.format(pb))
    else:
        equalvar = False
        print(u'  Variances are unequal (Bartletts test, p = %.3f); not assuming equal variances'.format(pb))
    (t, p) = Stats.ttest_ind(g1, g2, equal_var=False)
    g1mean = np.mean(g1)
    g1std = np.std(g1)
    g2mean = np.mean(g2)
    g2std = np.std(g2)
    #       df = (tstd[k]**2/tN[k] + dstd[k]**2/dN[k])**2 / (( (tstd[k]**2 /
    # tN[k])**2 / (tN[k] - 1) ) + ( (dstd[k]**2 / dN[k])**2 / (tN[k] - 1) ) )
    df = (g1std**2/n1 + g2std**2/n2)**2 / (((g1std**2 / n1)**2 / (n1 - 1) + ((g2std**2 / n2)**2 / (n1 - 1))))
    if dataLabel is not None:
        n = max([len(l) for l in k])
        print 'T-test, data set = %s' % (dataLabel)
        print(u'  {:s}={:8.{pc}f}\u00B1{:.{pc}f}  (mean, SD)'.format(k[0].rjust(n), g1mean, g1std, pc=decimals))
        print(u'  {:s}={:8.{pc}f}\u00B1{:.{pc}f}  (mean, SD)'.format(k[1].rjust(n), g2mean, g2std, pc=decimals))
        print(u'  t({:6.2f})={:8.4f}   p={:8.6f}'.format(df, float(t), float(p)))
    return(df, float(t), float(p))

def ranksums(data, dataLabel=None, decimals=4):
    """
    perform a t-test using Scipy.stats
    :param data: dictionary {'g1': [data], 'g2': [data]}
    :param dataLabel: string with label to be printed for summary information
    :return df: degrees of freedom calculated assuming unequal variance
    :return t: t statistic for the difference
    :return p: p value
    """
    k = data.keys()
    g1 = data[k[0]]
    g2 = data[k[1]]
    n1 = len(g1)
    n2 = len(g2)
    (z, p) = Stats.ranksums(g1, g2)
    g1mean = np.mean(g1)
    g1std = np.std(g1)
    g2mean = np.mean(g2)
    g2std = np.std(g2)
    (w1, p1) = Stats.shapiro(g1, a=None, reta=False)
    (w2, p2) = Stats.shapiro(g2, a=None, reta=False)
    if p1 > 0.05 or p2 > 0.05:
        print(u'Some data appear normally distributed: Shapiro-Wilk Group 1 p = {:6.3f}, Group2 p = {:6.3f}'.format(p1, p2))
        print (u'    (Nonparametric does not assume normal distribution)')
    else:
        print(u'Both data sets are normally distributed by Shapiro-Wilk test; could use t-test')
    if dataLabel is not None:
        n = max([len(l) for l in k])
        print 'Wilcoxon Rank Sums test, data set = %s' % dataLabel
        print(u'  {:s}={:8.{pc}f}\u00B1{:.{pc}f} (mean, SD)'.format(k[0].rjust(n), g1mean, g1std, pc=decimals))
        print(u'  {:s}={:8.{pc}f}\u00B1{:.{pc}f} (mean, SD)'.format(k[1].rjust(n), g2mean, g2std, pc=decimals))
        summarizeData(data, decimals=decimals)
        # iqr1 = np.subtract(*np.percentile(g1, [75, 25]))
        # iqr2 = np.subtract(*np.percentile(g2, [75, 25]))
        # print(u'  {:s}: median={:8.4f}  IQR={:8.4f}'.format(k[0].rjust(n), np.median(g1), iqr1))
        # print(u'  {:s}: median={:8.4f}  IQR={:8.4f}'.format(k[1].rjust(n), np.median(g2), iqr2))
        print(u'   z={:8.4f}   p={:8.6f}'.format(float(z), float(p)))
    return(float(z), float(p))    


def summarizeData(data, dataLabel=None, decimals=4):
    """
    Provide descriptive median and interquartile range for non-normal (or small) data sets
    """
    if dataLabel is not None:
        print '%s: Data Set Summary (median, IQR)' % dataLabel
    n = max([len(l) for l in data.keys()])
    for i, k in enumerate(data.keys()):
        g1 = data[k]
        iqr1 = np.subtract(*np.percentile(g1, [75, 25]))
        print(u'  {:s}:  {:8.{pc}f}, {:.{pc}f} (median, IQR)'.format(k.rjust(n), np.median(g1), iqr1, pc=decimals))    


def test(ngroups=3):
    """
    Tests for anova routines
    3 group anova taken from Prism 6.01 example
    """
    data={'Control': [54, 23, 45, 54, 45, 47], 'Treated': [87, 98, 64, 77, 89], 
    'TreatedAntagonist': [45, 39, 51, 49, 50, 55]}
    OneWayAnova3(dataDict=data, dataLabel='3Groups', mode='parametric')
    print '-'*80
    print 'Compare to Prism output: '
    print """
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
    "  Number of values (total)"	17				"""
    print '-'*80
    print 'Multiple comparisions from Prism:'
    print """
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

    """
if __name__ == '__main__':
    test()
    