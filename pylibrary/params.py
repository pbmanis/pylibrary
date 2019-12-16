"""
Params.py
A clsss to create parameter structures
First version: Paul Manis on 2014-01-31.
Distributed under MIT/X11 license. See license.txt for more infofmation.
"""

from __future__ import print_function


class Params(object):
    """
    utility class to create parameter lists...
    create like: p = Params(abc=2.0, defg = 3.0, lunch='sandwich')
    reference like p.abc, p.defg, etc.
    Supports getting the keys, finding whether a key exists, returning
    the strucure as a simple dictionary, and printing (show) the
    parameter structure.
    """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def getkeys(self):
        """
        Get the keys in the current dictionary
        """
        return self.__dict__.keys()

    def haskey(self, key):
        """
        Find out if the param list has a specific key in it
        """
        return key in self.__dict__.keys()

    def todict(self):
        """
        convert param list to standard dictionary
        Useful when writing the data
        """
        res = {}
        for dictelement in list(self.__dict__):
            if isinstance(self.__dict__[dictelement], Params):
                #print 'nested: ', dictelement
                res[dictelement] = self.__dict__[dictelement].todict()
            else:
                res[dictelement] = self.__dict__[dictelement]
        return res

    def show(self):
        """
        print the parameter block created in Parameter Init
        """
        print("--------    Parameter Block    ----------")
        for key in list(self.__dict__.keys()):
            print("%15s = " % (key), self.__dict__[key]) # eval('self.%s' % key)
        print("-------- ---------------------- ----------")


class ParamTests(object):
    """
    Perform some tests (not implemented)
    Also illustrates how to use this module
    """
    def __init__(self):
        self.testpar = Params(number=1, string='string', dict={'x': 0, 'y': 1})
        """
        Run a simple test to verify this works
        """

        print(self.testpar.getkeys())
        print(self.testpar.haskey('number'))
        print(self.testpar.haskey('notinkeys'))
        print(self.testpar.todict())
        self.testpar.show()
        import pickle
        with open('testparams.pkl', 'wb') as f:
            pickle.dump(self.testpar, f)
            
        with open('testparams.pkl', 'rb') as f:
            self.testpar2 = pickle.load(f)
        
        


if __name__ == '__main__':
    ParamTests()

