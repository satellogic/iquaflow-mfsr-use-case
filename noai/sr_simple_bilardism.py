import cv2
import numpy as np
# from SR_utils import *
# from SR_align_tools import *

#################################################################
# from SR_utils import *

def zoom_x(img, z, interpolation=cv2.INTER_LINEAR):
    return img if z==1 else cv2.resize(img, (int(img.shape[1]*z), int(img.shape[0]*z)), interpolation=interpolation)

#################################################################
# from SR_align_tools import *

def diff(img1, img2):
    dif = img1.astype(np.float)[10:-10,10:-10] - img2[10:-10,10:-10]
    return np.abs(dif)

def dif_align(img1, img2):
    w= 5
    c= w//2
    difs= np.zeros((w,w))
    for i,j in ((i,j) for i in range(w) for j in range(w)):
        difs[i,j] = round(diff(img1[c:-c-1, c:-c-1], img2[i:i-w, j:j-w]).mean(), 2)
    return difs

def prom_imgs(im_l):
    prom = im_l[0].astype(np.int)
    for im in im_l[1:]:
        prom += im
    return (prom/len(im_l)).astype(np.uint8)

def realign(srcs, show=False, base="mean"):
    difs=[]
    ofs= []
    if base=="mean":
        imb = prom_imgs(srcs)
    else:
        imb = srcs[base]
    for im in srcs:
        dif_al = dif_align(imb[5:-5, 5:-5], im[5:-5, 5:-5])
        #print(dif_al)
        min_p = np.argmin(dif_al)
        c_y, c_x = min_p//5-2, min_p%5-2
        ofs.append((c_y, c_x, dif_al[c_y+2, c_x+2]))
        #print(c_y, c_x, dif_al[c_y+2, c_x+2])
        if show:
            print(dif_al)
            dif = diff(imb, im)
            difs.append(np.clip(dif*3,0,255))
    if show:
        show_img(collage(difs, int(np.ceil(len(srcs)/4)), border_light=127), figsize=(16,12))
        print(ofs)
    return ofs

def comp_w(srcs, wghs, b=3): # Compose list of images using list of weights
    slc = (slice(b,-b), slice(b,-b)) 
    wg= np.zeros_like(wghs[0][slc])
    comp = np.zeros_like(wghs[0][slc])
    for i in range(len(srcs)):
        comp+= srcs[i][slc] * wghs[i][slc]
        wg+= wghs[i][slc]
    comp = comp/wg
    return comp.astype(np.uint8), wg

def calc_hom_ecc(im1_gray, im2_gray, warp_mode=cv2.MOTION_HOMOGRAPHY, number_of_iterations=15):
    # Define the motion model
    # warp_mode = cv2.MOTION_TRANSLATION
    # warp_mode = cv2.MOTION_HOMOGRAPHY

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    if cv2.__version__[:3]=='4.1': # bug in cv2
        try:
                (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria, None, 3)
        except:
                (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
    else:
        try:
                (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
        except:
                (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria, None, 3)
    return (cc, warp_matrix)

def aply_warp(im, warp_matrix):
    return cv2.warpPerspective (im, warp_matrix, (im.shape[1],im.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

#################################################################

sharp_ker_5x5 = np.array([[.0001, .001, .002, .001, .0001],
                          [.001, .004, .006, .004, .001],
                          [.002, .006,   1,  .006, .002], 
                          [.001, .004, .006, .004, .001],
                          [.0001, .001, .002, .001, .0001]])


def sr_bilardism(imgs, zoom = 3 , interpolation = cv2.INTER_LINEAR):
    # zoom: only 3 or 5  # interpolation: cv2.INTER_LINEAR / INTER_AREA ...
    ker = sharp_ker_5x5[1:-1, 1:-1] if zoom==3 else sharp_ker_5x5
    w_ori = np.tile(ker, imgs[0].shape)
    zoom_imgs = [zoom_x(img, zoom, interpolation=interpolation) for img in imgs]
    ws    = [w_ori for i in range(len(imgs))]
    ofs = realign(zoom_imgs)
    comp, wghs = comp_w(zoom_imgs, ws, b=5)
    return comp

def sr_warp_weights(imgs, zoom= 3, interpolation= cv2.INTER_LINEAR):
    if zoom==1.5:
        upsample = 3  # warp_weights upsample on 3 or 5 factor only
        subsample = 2
    else:
        upsample = zoom
        subsample = 1
    # upsample: only 3 or 5  # interpolation: cv2.INTER_LINEAR / INTER_AREA ...
    ker = sharp_ker_5x5[1:-1, 1:-1] if upsample==3 else sharp_ker_5x5
    w_ori = np.tile(ker, imgs[0].shape)
    aligs = [zoom_x(imgs[0], upsample, interpolation=interpolation)]
    ws    = [w_ori]
    for crop in imgs[1:]:
        im = zoom_x(crop, upsample, interpolation=interpolation)
        cc, warp_matrix = calc_hom_ecc(aligs[0], im)
        ali = aply_warp(im, warp_matrix)
        aligs.append(ali)
        ws.append(aply_warp(w_ori.copy(), warp_matrix))
    comp, wghs = comp_w(aligs, ws, b= 6)
    #print(imgs[0].shape, comp.shape)
    return zoom_x(comp, 1/subsample)
