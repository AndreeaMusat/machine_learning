#!/usr/bin/python
# -*- coding: utf-8 -*-
# Tudor Berariu, 2016

from pylab import plot, show, arange
from radius import radius

x = arange(1, 1001, 1)
y = [radius(i, 1000, 20, 20) for i in x]
plot(x,y)
show()
