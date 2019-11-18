# /bin/env python

""" A random collection of classes, collections, etc... that may be usefull in various applications
"""

from math import sqrt, floor
import pdb

def triangular_number(n: int) -> int:
    """ Caclulate the triangular number (n*(n+1)/2 """
    return n*(n+1)//2

def rational_to_count(x: int,y: int) -> int:
    """ Convert a rational number (with both counter and enumerator positive) to a unique id.
        this is possible rationals are countable. The id returned is the count.
    """
    # Assume the x-axis represents the counter and the Y-axis represents the enumerator.
    # Draw lines with slope -1 through each number on the X-axis
    # Starting from (0,0), count along these lines starting at the X-axis up the the Y-axis. Then jump to the next line.
    # You'll find that each (X,0) corresponds to one of the triangular numbers X*(X+1)/2
    # Some simple math gives then the below formula for (X,Y)
    return triangular_number(x+y) + y


def count_to_rational(count: int) -> (int,int):
    """ Convert a natural number to a unique rational
    """
    # This is much more difficult, as it requires reversing triangular numbers,
    # For any number, you are looking for the largest triangular number less than count. n*(n+1)/ <= count
    # => n^2 + n - 2*count <= 0  && solve for n. This is the X-intersect
    # knowing the slope of the lines, you can now find Y and X
    X_intersect = int((sqrt(8*count+1)-1)/2)
    base = triangular_number(X_intersect)
    Y = count - base
    X = X_intersect - Y
    return (X,Y)

if __name__ == '__main__':
    assert(count_to_rational(6) == (3,0))
    assert(count_to_rational(14) == (0,4))
    assert(count_to_rational(18) == (2,3))

