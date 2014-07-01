#!/usr/bin/env python
# encoding: utf-8
"""
RStats.py
Wrappers for using R for data anlsysis
Currently implements 1-way ANOVA for 3, 4, and 5 groups with all-way posttests
Created by Paul Manis on 2014-06-26.

"""
import scipy.stats

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


def OneWayAnova3(dataDict=None, dataLabel='data', mode='parametric'):
    """
    Perform a one way ANOVA with 3 groups
    :param dataDict: Dictionary holding data in format
        {'group1': [dataset], 'group2': [dataset], 'group3': [dataset]}
    :param dataLabel: title to use for print out of data
    :param mode: 'parametric' (default) uses R aov. anything else uses Kruskal-Wallis
    Prints resutls and post-hoc tests...
    :return: Nothing
    """ 
    labels = dataDict.keys()
    data = [[]]*len(labels)
    for i, d in enumerate(labels):
        data[i] = dataDict[d]
    if len(data) < 3:
        print "Need 3 groups to run One Way Anova here"
        return
    print 'OneWayAnova for %s' % (dataLabel)
    da1=dataDict[labels[0]]
    da2=dataDict[labels[1]]
    da3=dataDict[labels[2]]
    (F, p) = scipy.stats.f_oneway(da1, da2, da3)
    print '\nOne-way Anova (Scipy.stats): F=%f, p = %8.4f'% (F, p)
    if R_imported is False:
        print 'R not found, skipping'
        return
    print '\nR yields (anova3): '
    g1 = FloatVector(da1)
    g2 = FloatVector(da2)
    g3 = FloatVector(da3)
    robjects.globalenv['g1'] = g1
    robjects.globalenv['g2'] = g2
    robjects.globalenv['g3'] = g3
    #print da1
    #print dir(da1)
    robjects.globalenv['d1'] = FloatVector(da1)
    robjects.globalenv['d2'] = FloatVector(da2)
    robjects.globalenv['d3'] = FloatVector(da3)
    robjects.globalenv['d3'] = da3
    robjects.globalenv['l1'] = labels[0]
    robjects.globalenv['l2'] = labels[1]
    robjects.globalenv['l3'] = labels[2]
    x=robjects.r("Yd <- c(g1, g2, g3)") # could do inside call, but here is convenient for testing values
    vData = robjects.r("""vData <- data.frame( Yd, 
                Group=factor(rep(c(l1, l2, l3), times=c(length(g1), length(g2), length(g3)))))""")
# OLD code for blanaced groups:
    #groups = RBase.gl(3, len(d3), len(d3)*3, labels=labels)
    #robjects.globalenv["vData"] = vData
    #weight = g1 + g2 + g3 # !!! Order is important here, get it right!
   # robjects.globalenv["weight"] = weight
   # robjects.globalenv["group"] = groups

    if mode == 'parametric':
        aov = robjects.r("aov(Yd ~ Group, data=vData)")
        print aov
        robjects.globalenv["aov"] = aov
        sum = robjects.r("summary(aov)")
        print sum
        drop = robjects.r('drop1(aov, ~., test="F")')
        print drop
        #x=RStats.anova(aov)
        #print x
        print 'Post Hoc (Tukey HSD): '
        y=robjects.r("TukeyHSD(aov)")
        print y
    else:
        kw = robjects.r("kruskal.test(Yd ~ Group, data=vData)")
        print kw
        loc = robjects.r('Group=factor(rep(c(l1, l2, l3), times=c(length(g1), length(g2), length(g3))))')
        kwmc = robjects.r('pairwise.wilcox.test(Yd, Group, p.adj="bonferroni")')
        print "PostTest: pairwise wilcox\n", kwmc
    #rprint = robjects.globalenv.get("print")
    # the boxplot is redundant, but worth checking out
    #bp = robjects.r("boxplot(Yd ~ Group, data=vData)")
    #rprint(bp)
#        import time
#        time.sleep(10)


def OneWayAnova4(dataDict=None, dataLabel='data', mode='parametric'):
    """
    Perform a one way ANOVA with 4 groups
    :param dataDict: Dictionary holding data in format
        {'group1': [dataset], 'group2': [dataset], 'group3': [dataset]}
    :param dataLabel: title to use for print out of data
    :param mode: 'parametric' (default) uses R aov. anything else uses Kruskal-Wallis
    Prints resutls and post-hoc tests...
    :return: Nothing
    """ 
    labels = dataDict.keys()
    data = [[]]*len(labels)
    for i, d in enumerate(labels):
        data[i] = dataDict[d]
    if len(data) < 4:
        print "Need 4 groups to run One Way Anova 4 here"
        return
    print 'OneWayAnova for %s' % (dataLabel)
    d1=dataDict[labels[0]]
    d2=dataDict[labels[1]]
    d3=dataDict[labels[2]]
    d4=dataDict[labels[3]]
    (F, p) = scipy.stats.f_oneway(d1, d2, d3, d4)
    print '\nOne-way Anova (Scipy.stats), for check: F=%f, p = %8.4f'% (F, p)
    if R_imported is False:
        'R not found, skipping'
        return
    print '\nR yields: ' 
    g1 = FloatVector(d1)
    g2 = FloatVector(d2)
    g3 = FloatVector(d3)
    g4 = FloatVector(d4)
    robjects.globalenv['g1'] = g1
    robjects.globalenv['g2'] = g2
    robjects.globalenv['g3'] = g3
    robjects.globalenv['g4'] = g4
    robjects.globalenv['d1'] = d1
    robjects.globalenv['d2'] = d2
    robjects.globalenv['d3'] = d3
    robjects.globalenv['d4'] = d4
    robjects.globalenv['l1'] = labels[0]
    robjects.globalenv['l2'] = labels[1]
    robjects.globalenv['l3'] = labels[2]
    robjects.globalenv['l4'] = labels[3]
    x=robjects.r("Yd <- c(g1, g2, g3, g4)") # could do inside call, but here is convenient for testing values
    vData = robjects.r("""vData <- data.frame( Yd, 
                Equation=factor(rep(c(l1, l2, l3, l4), times=c(length(g1), length(g2), length(g3), length(g4)))))""")
# OLD code for blanaced groups:
    #groups = RBase.gl(3, len(d3), len(d3)*3, labels=labels)
    #robjects.globalenv["vData"] = vData
    #weight = g1 + g2 + g3 # !!! Order is important here, get it right!
   # robjects.globalenv["weight"] = weight
   # robjects.globalenv["group"] = groups

    if mode == 'parametric':
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
        loc = robjects.r('Equation=factor(rep(c(l1, l2, l3, l4), times=c(length(g1), length(g2), length(g3), length(g4))))')
        kwmc = robjects.r('pairwise.wilcox.test(Yd, Equation, p.adj="bonferroni")')
        print "PostTest: pairwise wilcox\n", kwmc
    #rprint = robjects.globalenv.get("print")
    # the boxplot is redundant, but worth checking out
    #bp = robjects.r("boxplot(Yd ~ Group, data=vData)")
    #rprint(bp)
#        import time
#        time.sleep(10)

def OneWayAnova5(dataDict=None, dataLabel='data', mode='parametric'):
    """
    Perform a one way ANOVA with 5 groups
    :param dataDict: Dictionary holding data in format
        {'group1': [dataset], 'group2': [dataset], 'group3': [dataset]}
    :param dataLabel: title to use for print out of data
    :param mode: 'parametric' (default) uses R aov. anything else uses Kruskal-Wallis
    Prints resutls and post-hoc tests...
    :return: Nothing
    """ 
    labels = dataDict.keys()
    data = [[]]*len(labels)
    for i, d in enumerate(labels):
        data[i] = dataDict[d]
    if len(data) < 4:
        print "Need 4 groups to run One Way Anova 4 here"
        return
    print 'OneWayAnova for %s' % (dataLabel)
    d1=dataDict[labels[0]]
    d2=dataDict[labels[1]]
    d3=dataDict[labels[2]]
    d4=dataDict[labels[3]]
    d5=dataDict[labels[4]]
    (F, p) = scipy.stats.f_oneway(d1, d2, d3, d4, d5)
    print '\nOne-way Anova (Scipy.stats), for check: F=%f, p = %8.4f'% (F, p)
    if R_imported is False:
        'R not found, skipping'
        return
    print '\nR yields: ' 
    g1 = FloatVector(d1)
    g2 = FloatVector(d2)
    g3 = FloatVector(d3)
    g4 = FloatVector(d4)
    g5 = FloatVector(d5)
    robjects.globalenv['g1'] = g1
    robjects.globalenv['g2'] = g2
    robjects.globalenv['g3'] = g3
    robjects.globalenv['g4'] = g4
    robjects.globalenv['g5'] = g5
    robjects.globalenv['d1'] = d1
    robjects.globalenv['d2'] = d2
    robjects.globalenv['d3'] = d3
    robjects.globalenv['d4'] = d4
    robjects.globalenv['d5'] = d5
    robjects.globalenv['l1'] = labels[0]
    robjects.globalenv['l2'] = labels[1]
    robjects.globalenv['l3'] = labels[2]
    robjects.globalenv['l4'] = labels[3]
    robjects.globalenv['l5'] = labels[4]
    x=robjects.r("Yd <- c(g1, g2, g3, g4, g5)") # could do inside call, but here is convenient for testing values
    vData = robjects.r("""vData <- data.frame( Yd, 
                Equation=factor(rep(c(l1, l2, l3, l4, l5), times=c(length(g1), length(g2), length(g3), length(g4), length(g5)))))""")
# OLD code for blanaced groups:
    #groups = RBase.gl(3, len(d3), len(d3)*3, labels=labels)
    #robjects.globalenv["vData"] = vData
    #weight = g1 + g2 + g3 # !!! Order is important here, get it right!
   # robjects.globalenv["weight"] = weight
   # robjects.globalenv["group"] = groups

    if mode == 'parametric':
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
        loc = robjects.r('Equation=factor(rep(c(l1, l2, l3, l4, l5), times=c(length(g1), length(g2), length(g3), length(g4), length(g5))))')
        kwmc = robjects.r('pairwise.wilcox.test(Yd, Equation, p.adj="bonferroni")')
        print "PostTest: pairwise wilcox\n", kwmc
    #rprint = robjects.globalenv.get("print")
    # the boxplot is redundant, but worth checking out
    #bp = robjects.r("boxplot(Yd ~ Group, data=vData)")
    #rprint(bp)
#        import time
#        time.sleep(10)

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
    