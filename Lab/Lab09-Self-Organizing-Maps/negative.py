#!/usr/bin/python
# -*- coding: utf-8 -*-
# Tudor Berariu, 2016

from PIL import Image
import copy
import sys

def negative(orig_file_name):
    # Argumentele: fișierul original

    orig_img = Image.open(orig_file_name)                            # Open file
    orig_pixels = list(orig_img.getdata())                          # Get pixels
    orig_pixels = [(o[0]/255.0, o[1]/255.0, o[2]/255.0) for o in orig_pixels]

    # orig_pixels conține (width x height) tupluri de 3 valori între 0.0 și 1.0
    neg_pixels = copy.copy(orig_pixels)

    ## Exercițiul 1: negativul imaginii originale
    ## Exercițiul 1: completați aici
    neg_pixels = [(1.0 - i[0], \
                   1.0 - i[1], \
                   1.0 - i[2]) for i in neg_pixels]

    ## ----------

    neg_pixels = [(int(o[0] * 255.0), int(o[1] * 255.0), int(o[2] * 255.0))
                     for o in neg_pixels]

    neg_img = Image.new('RGB', orig_img.size)
    neg_img.putdata(neg_pixels)
    neg_img.show()

    neg_img.save(orig_file_name.replace(".", "_neg."))

if __name__ == "__main__":
    negative(sys.argv[1])
