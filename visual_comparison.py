import os
import shutil
import cv2

import numpy as np
import matplotlib.pyplot as plt

from glob import glob

#########################
# Visual comparison
#########################

def scatter_plots(df):

    for met1,met2 in zip(
        [
            'ssim',
            'gmsd',
            'fid',
            'rer_0'
        ],
        [
            'psnr',
            'mdsi',
            'swd',
            'snr'
        ]
    ):

        fig, ax = plt.subplots()

        marker_lst = []

        for i in df.index:

            if 'hrn' in df['ds_modifier'][i]:
                marker = '*'
            elif 'msrn' in df['ds_modifier'][i]:
                marker = 'X'
            else:
                marker = 'o'

            ax.scatter(
                df[met1][i],
                df[met2][i],
                s=250.,
                marker=marker,
                label=df['ds_modifier'][i],
                alpha=0.5,
                edgecolors='none'
            )

        ax.set_xlabel(('rer' if 'rer_0'==met1 else met1))
        ax.set_ylabel(('rer' if 'rer_0'==met2 else met2))
        ax.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)

        plt.show()

def visual_comp():
    
    lst_lst = [
        glob(r"./xviewds/xview-ds/test#mfsr+agk_modifier/L0/*"),
        glob(r"./xviewds/xview-ds/test#mfsr+warpw_modifier/L0/*"),
        glob(r"./xviewds/xview-ds/test#mfsr+msrn_modifier/L0/*"),
        glob(r"./xviewds/xview-ds/test#mfsr+hrn_hrn_exp27_exp27-HRNet_30_modifier/L0/*")
    ]

    title_lst = ['AGK','WARPW','MSRN','HRN']
    str_fill = "*****"

    for enu,fn in enumerate( lst_lst[0] ):

        print(f'SAMPLE {enu} '+'*'*175)

        if enu>100:
            break

        n_alg = len(lst_lst)

        fn_lst = [
            glob(os.path.join(
                os.path.dirname(lst_lst[i][0]),
                os.path.basename(fn).split('+')[0]+'*.png'
            ))[0]
            for i in range( n_alg )
        ]

        arr_lst = [ cv2.imread( fn_lst[i] ) for i in range( n_alg ) ]

        vmin=np.min([el.min() for el in arr_lst])
        vmax=np.max([el.max() for el in arr_lst])

        fig,ax = plt.subplots(1, n_alg ,figsize=(27,14), gridspec_kw={'wspace':0, 'hspace':0},squeeze=True)
        for i in range( n_alg ):
            ax[i].imshow( arr_lst[i][50:-50:,50:-50:] , cmap='gray')
            ax[i].axis('off')
            ax[i].set_title(title_lst[i])

        fig,ax = plt.subplots(1, n_alg ,figsize=(27,14), gridspec_kw={'wspace':0, 'hspace':0},squeeze=True)
        for i in range( n_alg ):
            ax[i].imshow( arr_lst[i][140:-140:,140:-140:] , cmap='gray')
            ax[i].axis('off')
            ax[i].set_title(title_lst[i])

        plt.show()