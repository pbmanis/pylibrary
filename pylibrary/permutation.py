import numpy as np
import numpy.random as npr

def permutation_resampling(case, control, num_samples, statistic):
    """Returns p-value that statistic for case is different
    from statistc for control."""

    observed_diff = abs(statistic(case) - statistic(control))
    num_case = len(case)

    combined = np.concatenate([case, control])
    diffs = []
    for i in range(num_samples):
        xs = npr.permutation(combined)
        diff = np.mean(xs[:num_case]) - np.mean(xs[num_case:])
        diffs.append(diff)

    pval = (np.sum(diffs > observed_diff) +
            np.sum(diffs < -observed_diff))/float(num_samples)
    return pval, observed_diff, diffs

if __name__ == '__main__':
    import pylab
# make up some data
    case = [94, 38, 23, 197, 99, 16, 141]
    control = [52, 10, 40, 104, 51, 27, 146, 30, 46]

    # find p-value by permutation resampling
    pval, observed_diff, diffs = \
          permutation_resampling(case, control, 10000, np.mean)

    # make plots
    pylab.title('Empirical null distribution for differences in mean')
    pylab.hist(diffs, bins=100, histtype='step', normed=True)
    pylab.axvline(observed_diff, c='red', label='diff')
    pylab.axvline(-observed_diff, c='green', label='-diff')
    pylab.text(60, 0.01, 'p = %.3f' % pval, fontsize=16)
    pylab.legend()
    pylab.savefig('examples/permutation.png')
