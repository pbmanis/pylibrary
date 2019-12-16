#!/usr/bin/env python
#
#   bootstrap.py - resampling analysis for multiple comparisons
#   Copyright (C) 2011 Gian-Carlo Pascutto <gcp@sjeng.org>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as
#   published by the Free Software Foundation, either version 3 of the
#   License, or (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#   Rationale:
#
#   This programs performs a statistical analysis of multiple
#   treatments, comparing either a set of treatments against a first,
#   reference treatment, or comparing all treatments against each other.
#   It will calculate the probability that the observed differences can be
#   explained by random luck only, and notify when this is so low as to be
#   very unlikely.
#
#   It is robust towards data that is not normally distributed or has
#   differing variances, and can correct for the multiple comparisons
#   performed while maintaining good statistical power.
#
#   The main goal is to serve as an aid during algorithmic optimization
#   of software, particularly when very small improvements, or improvements
#   causing widely varying performance need to be tested. It has been
#   successfully used for applications ranging from psychoacoustic audio
#   encoders to artificial intelligence software.
#
#   This version is a complete rewrite of http://ff123.net/bootstrap
#   to address the shortcomings of the former.
#
#   Usage:
#
#   Provide the measurements in the form of a text file, with each treatment
#   being a separate column, and put each measurement on a row. Lines
#   starting with a % are considered to be comments. Example:
#
#   % test of rendering performance (fps)
#   ref   shader1 shader2 shader3
#   100     105     110     101
#    50      55      60      51
#    75      74      80      76
#   150     160     165     155
#
#   By default, all treatments are compared with the first. If you want to
#   compare all treatments among each other, specify the --compare-all option.
#
#   By default, the program assumes the measurements are random and not
#   related inside a row. If measurements on the same row are related, for
#   example because they represent the same input data, the same tester,
#   hardware, etc, then one should specify the --blocked option. In the
#   example data above, this is the case.
#
#   By default, the program will check the differences between the means
#   of each treatment. If the data is particularly nonstandard such that
#   comparing the means is misleading or meaningless, the comparison can be
#   based on the median instead. Note that one doesn't need to switch to the
#   median only because the data is not normally distributed: the used
#   algorithms are robust against this.
#
#   Note that because the program uses a Monte Carlo approximation during
#   the resampling analysis, the results are not exact and may vary slightly
#   from run to run.
#
#   Requirements:
#
#   Python v2.5 or later (not compatible with Python 3)
#
#   Version history:
#
#   v1.0 2011-02-03: Initial public release
#

REVISION = "v1.0 2011-02-03"

import optparse
import random


def transpose(data):
    """Transpose a 2-dimensional list."""
    return list(zip(*data))


def comparisons(labels, compareall):
    """Return a list of tuples representing the treatments to
       compare. If compareall is false, only comparisons against
       the first treatment are performed. If true, all comparisons
       are done."""
    if compareall:
        return [(x, y) for x in range(len(labels))
                for y in range(len(labels))
                if y < x]
    else:
        return [(x, 0) for x in range(len(labels)) if x != 0]


def means(data):
    """Return the list of means corresponding to the data passed
       in table form."""
    return [sum(x) / len(x) for x in transpose(data)]


def medians(data):
    """Return the list of medians corresponding to the data passed
       in table form. For an even number of items, the median is
       defined as the element just before the halfway point."""
    return [srow[len(srow) // 2] for srow in
            [sorted(row) for row in transpose(data)]]


def diffmean(xl, yl):
    """Return the difference between the means of 2 lists."""
    return abs(sum(xl) / len(xl) - sum(yl) / len(yl))


def variance(xl):
    """Return the variance of a list."""
    mean = sum(xl) / len(xl)
    return sum([(x - mean) ** 2 for x in xl]) / (len(xl) - 1)


def welcht(xl, yl):
    """Welch t-test for independant samples. This is similar
    to the student t-test, except that it does not require
    equal variances. Returns the test statistic."""
    # Denominator in the Welch's t statistic. This is a
    # variance estimate based on the available sample.
    # See also: Behrens-Fisher problem."""
    denom = (variance(xl) / len(xl) + variance(yl) / len(yl)) ** 0.5
    # Avoid division by zero if both lists are identical
    denom = denom + 1e-35
    return diffmean(xl, yl) / denom


def student_paired(xl, yl):
    """Student t-test for paired samples. Arguments are two lists of
    measurements. Returns the test statistic."""
    diffs = [x - y for x, y in zip(xl, yl)]
    meandiff = abs(sum(diffs) / len(diffs))
    ssd = variance(diffs) ** 0.5
    # Avoid division by zero if both lists are identical
    ssd = ssd + 1e-35
    return meandiff / (ssd * (len(diffs) ** 0.5))


def wilcoxon(xl, yl):
    """Performs a Wilcoxon signed-rank test for paired samples,
       with adjustments for handling zeroes due to Pratt. 
       The test statistic is inverted compared to a normal Wilcoxon 
       signed-rank test and cannot be interpreted directly."""
    diffs = [x - y for x, y in zip(xl, yl) if x != y]
    abs_diff = [abs(x) for x in diffs]
    abs_diff_rank = sorted((diff, idx) for idx, diff in enumerate(abs_diff))
    w_plus = 0
    w_minus = 0
    uniqrank = {}
    # ties at 0 increment the start rank (Pratt, 1959)
    startrank = len(xl) - len(diffs)
    for rank, (diff, idx) in enumerate(abs_diff_rank):
        if diff in uniqrank:
            rank = uniqrank[diff]
        else:
            # (start + (start + ties - 1)) / 2
            ties = abs_diff.count(diff)
            rank = (2 * (rank + 1) + (ties - 1)) / 2.0
            uniqrank[diff] = rank
        if diffs[idx] > 0:
            w_plus += startrank + rank
        else:
            w_minus += startrank + rank
    # invert by making high values more significant,
    # simplifies rest of code
    return 1.0 / (1.0 + min(w_plus, w_minus))


def mann_whitney(xl, yl):
    """Mann-Whitney-Wilcoxon U test for independant samples. This is
    the nonparametric alternative to the student t-test."""
    # make a merged list of all values, and sort it, but remember
    # from which sample each value came
    ranked = sorted([(x, 0) for x in xl] + [(y, 1) for y in yl])
    x_ranksum = 0
    uniqrank = {}
    for rank, (value, series) in enumerate(ranked):
        if value in uniqrank:
            rank = uniqrank[value]
        else:
            ties = ranked.count((value, 0)) + ranked.count((value, 1))
            rank = (2 * (rank + 1) + (ties - 1)) / 2.0
            uniqrank[value] = rank
        if series == 0:
            x_ranksum += rank
    u_1 = x_ranksum - ((len(xl) * (len(xl) + 1)) / 2)
    u_2 = (len(xl) * len(yl)) - u_1
    # invert to make higher more significant
    return 1.0 / (1.0 + min(u_1, u_2))


def get_stats(func, data, compars):
    """Apply the test statistic passed as 'func' to the 'data' for
       the pairs to compare in 'compars'."""
    tdata = transpose(data)
    return [func(tdata[x], tdata[y]) for x, y in compars]


def permute(data, blocked=False):
    """Perform a resampling WITHOUT replacement to the table in 'data'.
       If 'blocked' is true, resamplings only happen within rows.
       If it is false, they are done throughout the table."""
    if blocked:
        return [random.sample(row, len(row)) for row in data]
    else:
        flat = [x for row in data for x in row]
        permuted = random.sample(flat, len(flat))
        return [[permuted.pop() for _ in row] for row in data]


def bootstrap(data, blocked=False):
    """Perform a resampling WITH replacement to the table in 'data'.
       If 'blocked' is true, resamplings only happen within rows.
       If it is false, they are done throughout the table."""
    if blocked:
        return [[random.choice(row) for _ in row] for row in data]
    else:
        flat = [x for row in data for x in row]
        return [[random.choice(flat) for _ in row] for row in data]


def resample_pvals(data, compars, options):
    """Given a set of data and a test statistic in options.teststat,
       calculate the probability that the test statistic is as extreme
       as it is due to random chance, for the comparisons in 'compars'.
       The probability is calculated by a re-randomization permutation."""
    teststat = get_stats(options.teststat, data, compars)
    phits = [0 for _ in compars]

    for _ in range(options.permutes):
        pdata = options.resample_func(data, options.blocked)
        pteststat = get_stats(options.teststat, pdata, compars)
        phits = [phits[z] + int(teststat[z] <= pteststat[z])
                 for z in range(len(compars))]

    return [phit / float(options.permutes) for phit in phits]


def stepdown_adjust(data, compars, options):
    """Calculate a set of p-values for 'data', and adjust them for the 
    multiple comparisons in 'compars' being performed. The used algorithm 
    is a resampling based free step-down using the max-T algorithm 
    from Westfall & Young."""
    tstat = get_stats(options.teststat, data, compars)
    # sort the test statistics, but remember which comparison they came from
    tstat_help = [(tval, idx) for idx, tval in enumerate(tstat)]
    sortedt = sorted(tstat_help)
    phits = [0 for _ in compars]

    for _ in range(options.stepdown):
        bdata = options.resample_func(data, options.blocked)
        btstat = get_stats(options.teststat, bdata, compars)

        # free step-down using maxT
        maxt = -1
        for torg, idx in sortedt:
            maxt = max(btstat[idx], maxt)
            if (maxt >= torg):
                phits[idx] = phits[idx] + 1

    # the new p-value is the ratio with which such an extremal
    # statistic was observed in the resampled data
    new_pval = [phit / float(options.stepdown) for phit in phits]

    # ensure monotonicity of p-values
    maxp = 0.0
    for _, idx in reversed(sortedt):
        maxp = max(maxp, new_pval[idx])
        new_pval[idx] = maxp

    return new_pval


def display_attrib(attrib, labels):
    """Display the given attribute list with the given list of labels."""
    for lbl in labels:
        print("%8s " % lbl, end=' ')
    print()
    for val in attrib:
        print("%8.3f " % val, end=' ')
    print()
    print()


def display_pvals(pvals, attrib, compars, labels, threshold=0.05):
    """Construct the matrix of comparisons and print the p-value for 
    each comparison. Mark each value more significant than the treshold.
    attrib should contain the means or medians as appropriate."""
    print("         ", end=' ')
    # The valid comparisons have the shape of an upper-triangular matrix.
    # The first column would contain 1 element, but that is just the
    # first label compared to itself, so we skip it entirely.
    for lbl in labels[1:]:
        print("%-8s " % lbl, end=' ')
    print()
    for y, yl in enumerate(labels):
        # number x-labels and drop the first one
        xes = [x for x, _ in enumerate(labels) if x != 0]
        # check for empty rows and skip them
        if not any((x, y) in compars for x in xes):
            break
        # row with some real info, display it
        print("%-8s " % yl, end=' ')
        for x in xes:
            if (x, y) in compars:
                pv = pvals[compars.index((x, y))]
                if pv < threshold:
                    print("%1.3f*   " % pv, end=' ')
                else:
                    print("%1.3f    " % pv, end=' ')
            else:
                print("-        ", end=' ')
        print()
    print()
    for x, y in compars:
        if attrib[x] > attrib[y]:
            direction = "better"
        else:
            direction = "worse"
        pval = pvals[compars.index((x, y))]
        if pval < threshold:
            print("%s is %s than %s (p=%1.3f)" % (labels[x],
                                                  direction, labels[y], pval))
    print()


def main(options, filename):
    labels = None
    data = []
    print("Reading from: %s" % filename)
    infile = open(filename, "r")
    for line in infile:
        if line.startswith("%"):
            continue
        if not labels:
            labels = line.split()
            continue
        data.append([float(x) for x in line.split()])
    print("Read %d treatments, %d samples" % (len(transpose(data)), len(data)), end=' ')
    compars = comparisons(labels, options.compareall)
    print("=> %d comparisons" % len(compars))

    if options.bootstrap:
        options.resample_func = bootstrap
    else:
        options.resample_func = permute

    if options.median:
        if options.blocked:
            options.teststat = wilcoxon
        else:
            options.teststat = mann_whitney
        print("Medians:")
        attrib = medians(data)
    else:
        if options.blocked:
            options.teststat = student_paired
        else:
            options.teststat = welcht
        print("Means:")
        attrib = means(data)
    display_attrib(attrib, labels)

    print("Unadjusted p-values:")
    pvals = resample_pvals(data, compars, options)
    display_pvals(pvals, attrib, compars, labels, threshold=options.alpha)

    print("p-values adjusted for multiple comparison:")
    adj_pvals = stepdown_adjust(data, compars, options)
    display_pvals(adj_pvals, attrib, compars, labels,
                  threshold=options.alpha)

if __name__ == "__main__":
    print("bootstrap.py %s" % REVISION)
    print("Copyright (C) 2011 Gian-Carlo Pascutto <gcp@sjeng.org>")
    print("License Affero GPL version 3 or later", end=' ')
    print("<http://www.gnu.org/licenses/agpl.html>")
    print()
    usage = "usage: %prog options datafile"
    parser = optparse.OptionParser(usage)
    parser.add_option("-p", action="store", type="int",
                      dest="permutes", metavar="PERMUTATIONS",
                      help="Amount of p-value resamplings to run "
                      "[default: %default]")
    parser.add_option("-s", action="store", type="int",
                      dest="stepdown", metavar="STEPDOWNS",
                      help="Amount of step-down resamplings to run "
                      "[default: %default]")
    parser.add_option("-a", action="store", type="float",
                      dest="alpha", metavar="ALPHA",
                      help="Confidence level below which to report results "
                      "[default: %default]")
    parser.add_option("--medians", action="store_true", dest="median",
                      help="Use medians as the comparison metric")
    parser.add_option("--paired", "--blocked", action="store_true",
                      dest="blocked",
                      help="Use paired/blocked comparisons (use when input "
                           "data on the same row is related)")
    parser.add_option("--compare-all", action="store_true",
                      dest="compareall",
                      help="Perform all possible pairwise comparisons")
    parser.add_option("--bootstrap", action="store_true", dest="bootstrap",
                      help="Use bootstraps instead of permutations")
    parser.set_defaults(stepdown=10000, permutes=10000, alpha=0.05,
                        medians=False, blocked=False, compareall=False,
                        bootstrap=False)
    (options, args) = parser.parse_args()
    if not len(args) == 1:
        parser.print_help()
        exit(1)
    main(options, args[0])
