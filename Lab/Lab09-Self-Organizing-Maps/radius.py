# -*- coding: utf-8 -*-
# Tudor Berariu, 2016

def radius(iter_no, iter_count, height, width):
    """Calculează raza în funcție de iterația curentă,
    numărul total de iterații și dimensiunea rețelei"""
    ## Exercițiul 3: calculul razei în funcție de dimensiunile rețelei
    ##           și de iterația curentă Exercițiul 3: completați aici

    ## Exercițiul 3: ----------
    m = max(width, height) / 2
    radius = m - m * (1.0 * iter_no / iter_count)
    return radius
