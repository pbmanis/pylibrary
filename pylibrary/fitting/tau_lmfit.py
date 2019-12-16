#!/usr/bin/env python
# encoding: utf-8
"""
Params.py

Created by Paul Manis on 2013-12-16.
Copyright 2010-2014  Paul Manis
Distributed under MIT/X11 license. See license.txt for more infofmation.
"""

import sys
import os
import pickle
import numpy as np
import lmfit

def main():
    f_tf = open('Ihfit4_taufast_a.p', 'r')
    tf = f_tf.load()
    p = lmfit.Parameters()
    p.add_many(('dc', 0), ('a', 1), ('vh1', -40), ('t1', 100.), ('b', 1), ('vh2', -100), ('t2', 100.))
    mi = lmfit.minimize(taucurve, p)
    mi.leastsq()
    lm.printfuncs.report_fit(mi.params)
    ci = lmfit.conf_interval(mi)
    lmfit.printfuncs.report_ci(ci)
    
def taucurve(self, p, x, y=None, C=None, sumsq=True, weights=None):
    """
    HH-like description of activation/inactivation function
    'DC', 'a1', 'v1', 'k1', 'a2', 'v2', 'k2'
    """
    yd = p['dc'] + 1.0 / (p['a'] * numpy.exp((x + p['vh1']) / p['t1']) + p['b'] * numpy.exp(-(x + p['vh2']) / p['t2']))
    if y == None:
        return yd
    else:
        if sumsq is True:
            return numpy.sqrt(numpy.sum((y - yd) ** 2))
        else:
            return y - yd    

def boltzeval(self, p, x, y=None, C=None, sumsq=False, weights=None):
    yd = p['o'] + (p['a1'] - p['a2']) / (1.0 + numpy.exp((x - p['vh']) / p['k']))
    if y == None:
        return yd
    else:
        if sumsq is True:
            return numpy.sqrt(numpy.sum((y - yd) ** 2.0))
        else:
            return y - yd


if __name__ == '__main__':
    main()

