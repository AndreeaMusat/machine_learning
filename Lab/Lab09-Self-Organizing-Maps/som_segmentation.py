#!/usr/bin/python
# -*- coding: utf-8 -*-
# Tudor Berariu, 2016

from PIL import Image
import copy
import sys
from learning_rate import learning_rate
from radius import radius
from neighbourhood import neighbourhood
import sys
import numpy as np
from random import choice


def som_segmentation(orig_file_name, n):
    ## După rezolvarea Exercițiilor 2, 3 și 4
    ## în fișierele learning_rate.py, radius.py și neighbourhood.py
    ## rezolvați aici Exercițiile 5 și 6

    orig_img = Image.open(orig_file_name)
    orig_pixels = list(orig_img.getdata())
    orig_pixels = [(o[0]/255.0, o[1]/255.0, o[2]/255.0) for o in orig_pixels]

    ## Exercițiul 5: antrenarea rețelei Kohonen
    ## Exercițiul 5: completați aici:
    try:
        n = int(n)
    except:
        sys.exit(-1)

    W = np.random.random((n, n, 3))
    iter_count = 1000

    euclidean_dist = lambda p1, p2 : np.sqrt(((p1 - p2)**2).sum())

    for it in range(iter_count+1):
        x_i = np.array(choice(orig_pixels))
        all_dists = [[euclidean_dist(x_i, W[i, j]) for i in range(n)] for j in range(n)]
        all_dists = np.array(all_dists)
        w_z_idx = np.unravel_index(np.argmin(all_dists), all_dists.shape)
        print('w idx =', w_z_idx)
        w_z = W[w_z_idx]
        print('w = ', w_z)

        print('i=%d, itercount=%d' % (it, iter_count))

        crt_learning_rate = learning_rate(it, iter_count)
        print('learning rate=', crt_learning_rate)
        crt_radius = radius(it, iter_count, n, n)
        print('crt radius =', crt_radius)
        crt_neighs = neighbourhood(w_z_idx[0], w_z_idx[1], crt_radius, n, n)

        old_w = np.copy(W)

        for i in range(n):
            for j in range(n):
                for channel in range(3):
                    new_val = W[i, j, channel] + crt_learning_rate * crt_neighs[i][j] * (x_i[channel] - W[i, j, channel])
                    print('new val=', new_val)
                    W[i, j, channel] = new_val

        # if np.sum(np.abs(W - old_w)) <= 0.00000001:
            # print('BREAK after %d iterations' % it)
            # break


    ## Exercițiul 5: ----------

    ## Exercițiul 6: crearea imaginii segmentate pe baza ponderilor W
    ## Exercițiul 6: porniți de la codul din show_neg
    ## Exercițiul 6: completați aici:
    # colors = ['red', 'green', 'blue', 'yellow', 'black', 'white', 'olive', 'pink', 'navy']
   
    for t in range(len(orig_pixels)):

        crt_pixel = np.array(orig_pixels[t])
        all_dists = [[euclidean_dist(crt_pixel, W[i, j]) for i in range(n)] for j in range(n)]
        all_dists = np.array(all_dists)
        # print(all_dists)
        min_dist_neuron_idx = np.unravel_index(np.argmin(all_dists), all_dists.shape)

        red = int(255 * W[min_dist_neuron_idx[0], min_dist_neuron_idx[1], 0])
        green = int(255 * W[min_dist_neuron_idx[0], min_dist_neuron_idx[1], 1])
        blue = int(255 * W[min_dist_neuron_idx[0], min_dist_neuron_idx[1], 2])

        orig_pixels[t] = (red, green, blue)
        # orig_pixels[t] = (red, green, blue)
        print('ok pixel', t, 'val=', orig_pixels[t])

    segmented_img = Image.new('RGB', orig_img.size)
    segmented_img.putdata(orig_pixels)
    segmented_img.show()

    ## Exercițiul 6: ----------
    pass

if __name__ == "__main__":
    som_segmentation(sys.argv[1], sys.argv[2])
