#!/usr/bin/python
# -*- coding: utf-8 -*-
# Tudor Berariu, 2016

from pylab import plot, show, arange
from learning_rate import learning_rate

x = arange(1, 1001, 1)
y = [learning_rate(i, 1000) for i in x]
plot(x,y)
show()
