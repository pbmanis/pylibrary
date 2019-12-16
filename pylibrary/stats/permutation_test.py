import numpy as np
import numpy.random as npr
import scipy.stats as st
import pylab

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

def count_thr(data, thr):
    x = data[np.where(data <= thr)]
    return len(x)
    
if __name__ == '__main__':
    # make up some data


    # for normal data, or for a uniform distirbution,
    # generate Nc pairs of different random distributions, with sample size N
    # (I keep it small to simulate our experimental sample size).
    # The distributions differ by deltas (in units of sigma)
    # the distributions are normalized and centered - e.g., the control is at 0
    # and the sigma is 1.0 (equal variance assumption)
    # We then compare the results of the following tests (p values):
    # 1. premutation resampling
    # 2. standard "t-test"
    # 3. Wilcoxon rank 
    # 3. Kolmogorov-Smirnov test
    N = 24 # sample size
    Nc = 100 # number of distributions to test
    deltas = [0.5]
    thr = 0.05 # set threshold to check against. 
    p_perm  = np.zeros((len(deltas), Nc))
    t_test  = np.zeros((len(deltas), Nc))
    ks_test = np.zeros((len(deltas), Nc))
    w_test  = np.zeros((len(deltas), Nc))
    for nt in range(Nc):
        ctl = npr.uniform(0., 1., N)
        for j,d in enumerate(deltas):
            exp = npr.uniform(d, 1., N)
            # find p-value by permutation resampling
            pval, observed_diff, diffs = permutation_resampling(exp, ctl, 10000, np.mean)
            t, pval_t = st.ttest_ind(exp, ctl) # equal varaince t-test
            ks, pval_ks = st.ks_2samp(exp, ctl) # ks test with 2 samples.
            wrank, pval_w = st.wilcoxon(exp, ctl) 
            p_perm[j, nt] = pval
            t_test[j, nt] = pval_t
            ks_test[j, nt] = pval_ks
            w_test[j, nt] = pval_w
             
    print( p_perm)
    for i in range(len(deltas)):
        print('Delta: %.2f with n = %d' % (deltas[i], Nc))
        perm_sig = count_thr(p_perm[i,:], thr)
        t_sig = count_thr(t_test[i,:], thr)
        ks_sig = count_thr(ks_test[i,:], thr)
        w_sig = count_thr(w_test[i,:], thr)
        print('   perm:      %4d %7.5f  %7.5f' %( perm_sig, np.mean(p_perm[i,:]), np.std(p_perm[i,:])))
        print('   ttest:     %4d %7.5f  %7.5f' %(t_sig, np.mean(t_test[i,:]), np.std(t_test[i,:])))
        print('   wilcoxon:  %4d %7.5f  %7.5f' % (w_sig, np.mean(w_test[i,:]), np.std(w_test[i,:])))
        print('   kstest:    %4d %7.5f  %7.5f' % (ks_sig, np.mean(ks_test[i,:]), np.std(ks_test[i,:])))
    


