import argparse
import glob
import os
import tempfile

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def mod_img_parallel( lst ):
    
    dst, data_input, imgset_subdir, instance_of_cls = lst
    
    lr_fn_lst = glob.glob( os.path.join(data_input,imgset_subdir,'LR*.png') )
    
    if not len(lr_fn_lst):
        return
        
    try:
        imgp = instance_of_cls._mod_img( lr_fn_lst )
        cv.imwrite( os.path.join(dst, imgset_subdir+'+'+instance_of_cls.params['algo']+'.png'), imgp )
    except Exception as e:
        print(e)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Adaptive Kernels MFSR.')
    parser.add_argument('input_path', type=str,
                        help='Path to input frames.')
    parser.add_argument('-ei', '--ecc_iter', help='Number of ECC iterations.',
                        default=15, type=int)
    parser.add_argument('-nf', '--num_frames', help='Number of LR frames to process.',
                        default=None, type=int)
    parser.add_argument('-sf', '--starting_frame', help='First frame of LR frames to process.',
                        default=None, type=int)
    parser.add_argument('-bf', '--base_frame',
                        help='Base frames for LR grid alignment.',
                        default=None, type=str)
    # parser.add_argument('-bc', '--base_chann',
    #                     help='Base channel for interband alignment.',
    #                     choices=['red', 'green', 'blue'], default='red',
    #                     type=str)
    parser.add_argument('-sc', '--subcrop', nargs=4,
                        help='Subcrop in LR pixel units. Format is [xcen, ycen, width, height]',
                        default=[0, 0, 0, 0], type=int)
    parser.add_argument('-sr', '--sr_factor', help='SR factor.',
                        default=3., type=float)

    args = parser.parse_args()
    return args


def load_LR_frames(working_dir, input_path, subcrop=[0, 0, 0, 0], lr_name_filter='LR*.png', nf=None, sf=None):
    start_frame, end_frame = _resolve_frame_slice(nf, sf)

    if subcrop == [0, 0, 0, 0]:
        xmin = 0
        xmax = None
        ymin = 0
        ymax = None
    else:
        xmin = subcrop[0] - subcrop[2] // 2
        xmax = subcrop[0] + subcrop[2] // 2
        ymin = subcrop[1] - subcrop[3] // 2
        ymax = subcrop[1] + subcrop[3] // 2

    file_paths = sorted([path for path in glob.iglob(input_path + '/' + lr_name_filter)])

    frames = {path.split('/')[-1].split('.')[0]:
                    cv.imread(path, cv.IMREAD_GRAYSCALE)[ymin:ymax, xmin:xmax]
                    for path in file_paths[start_frame:end_frame]}

    n_frames = len(frames)

    print('\nInput LR frames:')
    for img in frames:
        print(img)
    print('Total number of input frames: ', n_frames)


    return frames, n_frames

def _resolve_frame_slice(nf, sf):
    if (sf and sf < 0) or (nf and nf < 0):
        raise NotImplementedError("Error: check start frame and number of frames")

    if not sf:
        sf = 0
    else:
        sf -= 1

    if nf:
        end_frame = sf + nf
    else:
        end_frame = None

    return sf, end_frame

def align_ecc(templateImage, inputImage, warp_mode=cv.MOTION_HOMOGRAPHY,
              border=0, number_of_iterations=15):
    im1 = templateImage
    im2 = inputImage

    # Find size of image1
    sz = im1.shape

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                number_of_iterations,  termination_eps)

    try:
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv.findTransformECC(im1[border:-border-1,
                                                    border:-border-1],
                                                im2[border:-border-1,
                                                    border:-border-1],
                                                warp_matrix, warp_mode, criteria)
    except:
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv.findTransformECC(im1[border:-border-1,
                                                    border:-border-1],
                                                im2[border:-border-1,
                                                    border:-border-1],
                                                warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)

    if warp_mode == cv.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv.warpPerspective(im2, warp_matrix, (sz[1], sz[0]),
                                         flags=cv.INTER_LINEAR +
                                         cv.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv.warpAffine(im2, warp_matrix, (sz[1], sz[0]),
                                    flags=cv.INTER_LINEAR +
                                    cv.WARP_INVERSE_MAP)

    return im2_aligned, warp_matrix


def frames_alignment(frames_dict, working_dir, base_frame_index, number_of_iterations):
    
    template = list(frames_dict.keys())[base_frame_index]
    lr_size = frames_dict[template].shape
    print('Template: ', template, '\tLR_size: ', lr_size)
    frames_dict_aligned = {}
    warp_matrices = {}

    for img in frames_dict:
        if img == template:
            (frames_dict_aligned[img],
             warp_matrices[img]) = (frames_dict[img],
                                    np.eye(3, 3,
                                           dtype=np.float32))
        else:
            (frames_dict_aligned[img],
             warp_matrices[img]) = align_ecc(frames_dict[template],
                                             frames_dict[img], border=2,
                                             number_of_iterations=number_of_iterations)
    print('Done!')
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        for img in frames_dict_aligned:
            cv.imwrite(
                os.path.join( tmp_dir, img + '_ecc_aligned.png'),
                frames_dict_aligned[img])
            
    #     if not os.path.isdir(working_dir + '/adaptive_kernels_byproducts'):
    #         os.mkdir(working_dir + '/adaptive_kernels_byproducts')
    #     for img in frames_dict_aligned:
    #         cv.imwrite(working_dir + '/adaptive_kernels_byproducts/' + img +
    #                    '_ecc_aligned.png', frames_dict_aligned[img])

    return warp_matrices, lr_size


def compute_homographies(LR_frames_dict, working_dir, base_frame_indices, number_of_iterations):
    print('\nAligning frames...')
    warp_matrices, lr_size = frames_alignment(LR_frames_dict,
                                              working_dir,
                                              base_frame_indices,
                                              number_of_iterations)

    return warp_matrices, lr_size


def hom_transf(H, x, y):
    npos = np.matmul(H, [x, y, 1])
    return npos[0]/npos[2], npos[1]/npos[2]


# def plot_base_grids(dewarp_grids, channel, working_dir, base_frame_index):
#     disp_width = 15
#     disp_height = 10
#     fig, ax = plt.subplots(figsize=(disp_width, disp_height))

#     for k, grid in enumerate(dewarp_grids):
#         u = grid[:, :, 0]
#         v = grid[:, :, 1]
#         if k == base_frame_index:
#             c = channel
#             s = 70
#         else:
#             c = 'black'
#             s = 20
#         plt.scatter(u, v, marker='o', c=c, s=s)
#     plt.xticks(np.arange(0, disp_width, step=1))
#     plt.yticks(np.arange(0, disp_height, step=1))
#     ax.axis('equal')
#     plt.xlim(-.2, disp_width-1+.2)
#     plt.ylim(-.2, disp_height-1+.2)
#     ax.invert_yaxis()
#     plt.grid()
#     plt.title('Dewarp grids')
#     plt.savefig(working_dir + '/adaptive_kernels_byproducts/dewarp_grids.png')


def transform_base_grids(warp_matrices, lr_size, n_frames, working_dir, base_frame_indices):
    print('\nDewarping base grid...')
    width = lr_size[1]
    height = lr_size[0]
    dewarp_grids = np.zeros((n_frames, ) + lr_size[::-1] + (2, ))

    for k, frame in enumerate(warp_matrices):
        for y in range(height):
            for x in range(width):
                dewarp_grids[k, x, y, :] = hom_transf(np.linalg.inv(warp_matrices[frame]), x, y)
    print('Done!')
    #plot_base_grids(dewarp_grids, 'magenta', working_dir, base_frame_indices)

    return dewarp_grids


def frame2index(frames_dict, base_frame):
    frames_names = list(frames_dict.keys())
    base_frame_index = frames_names.index(
        base_frame) if base_frame in frames_names else 0

    return base_frame_index


def gather_win_points_from_table(dewarp_table, x, y, win_size=3, search_offset=5):
    x1 = np.clip(int(round(x - (search_offset+win_size/2))), 0, dewarp_table.shape[1])
    x2 = np.clip(int(round(x + (search_offset+win_size/2))), 0, dewarp_table.shape[1])
    y1 = np.clip(int(round(y - (search_offset+win_size/2))), 0, dewarp_table.shape[2])
    y2 = np.clip(int(round(y + (search_offset+win_size/2))), 0, dewarp_table.shape[2])

    w = np.where((np.abs(x-dewarp_table[:, x1:x2, y1:y2, 0]) < win_size/2) &
                 (np.abs(y-dewarp_table[:, x1:x2, y1:y2, 1]) < win_size/2))
    w = np.stack(w).transpose() + [0, x1, y1]
    return w


def recons_rect(dewarp_grids, img_table, lr_size, sr_factor, base_frame_index):
    win_size = .5
    lonely_px_factor = 2
    offsets = dewarp_grids - dewarp_grids[base_frame_index]
    search_offset = np.ceil(np.max(offsets))

    xmin = 0
    xmax = lr_size[1]
    ymin = 0
    ymax = lr_size[0]

    sr_img_ = np.zeros(np.floor(sr_factor*np.array([ymax-ymin, xmax-xmin])).astype(np.int))

    for n, x in enumerate(np.linspace(xmin, xmax-1, num=int(sr_factor*(xmax-xmin)) )):
        for m, y in enumerate(np.linspace(ymin, ymax-1, num=int(sr_factor*(ymax-ymin)) )):
            win_points = gather_win_points_from_table(dewarp_grids, x, y, win_size=win_size,
                                                      search_offset=search_offset)
            # sr_pixels = [img_table[p[0], p[2], p[1]] for p in win_points]
            sr_pixels = img_table[(win_points[:,0], win_points[:,2], win_points[:,1])]

            if sr_pixels.size != 0:
                sr_img_[m, n] = np.mean(sr_pixels)
            else:
                win_points = gather_win_points_from_table(dewarp_grids, x, y,
                                                          win_size=lonely_px_factor*win_size,
                                                          search_offset=search_offset)
                # sr_pixels = [img_table[p[0], p[2], p[1]] for p in win_points]
                sr_pixels = img_table[(win_points[:,0], win_points[:,2], win_points[:,1])]
                sr_img_[m, n] = np.mean(sr_pixels) if sr_pixels.size != 0 else 0.0
        print('\033[FProgress: ', round(float(n+1) / float(sr_factor*(xmax-xmin)) * 100), '%')

    return np.clip(sr_img_, 0, 255).astype(np.uint8)


def build_img_tables(frames_dict, lr_size, n_frames):
    img_tables = np.zeros((n_frames, ) + lr_size, dtype=np.uint8)
    for k, frame in enumerate(frames_dict):
        img_tables[k, :, :] = frames_dict[frame]

    return img_tables


def rect_ker_SR(dewarp_grids, img_tables, lr_size, sr_factor, base_frame_indices):
    print('\nRectangular kernel reconstruction')
    print('Reconstructing...\n')
    sr_imgs_rect_ker = recons_rect(dewarp_grids, img_tables, lr_size,
                                          sr_factor, base_frame_indices)
    print('Done!')

    return sr_imgs_rect_ker


def recons_adap(dewarp_grids, img_table, cov_matrix, lr_size, sr_factor, base_frame_index):
    win_size = 1.5
    offsets = dewarp_grids - dewarp_grids[base_frame_index]
    search_offset = np.ceil(np.max(offsets))

    xmin = 0
    xmax = lr_size[1]
    ymin = 0
    ymax = lr_size[0]

    sr_img_ = np.zeros(np.floor(sr_factor*np.array([ymax-ymin, xmax-xmin])).astype(np.int))

    for n, x in enumerate(np.linspace(xmin, xmax-1, num=int(sr_factor*(xmax-xmin)))):
        for m, y in enumerate(np.linspace(ymin, ymax-1, num=int(sr_factor*(ymax-ymin)))):
            inv_cov = np.linalg.inv(cov_matrix[m, n])
            win_points = gather_win_points_from_table(dewarp_grids, x, y, win_size=win_size,
                                                      search_offset=search_offset)
            # sr_pixels = [img_table[p[0], p[2], p[1]] for p in win_points]
            sr_pixels = img_table[(win_points[:,0], win_points[:,2], win_points[:,1])]

            # distances = np.array([dewarp_grids[tuple(p)] - [x, y] for p in win_points])
            distances = dewarp_grids[tuple(win_points.transpose())] - [x, y]
            # coef = [np.exp(-.5*np.transpose(d).dot(np.linalg.inv(cov_matrix[m, n])).dot(d))
            #         for d in distances]
            coef = np.exp(-.5*np.multiply(distances.dot(inv_cov), distances).sum(1))
            sr_img_[m, n] = np.average(sr_pixels, weights=coef) if sr_pixels.size != 0 else 0.0
        print('\033[FProgress: ', round(float(n+1) / float(sr_factor*(xmax-xmin)) * 100), '%')

    return np.clip(sr_img_, 0, 255).astype(np.uint8)


def adap_ker_SR(dewarp_grids, img_tables, cov_matrices, lr_size, sr_factor, base_frame_indices):
    print('\nAdaptive Gaussian kernel reconstruction')
    print('Reconstructing...\n')
    sr_imgs_adap_ker = recons_adap(dewarp_grids, img_tables, cov_matrices,
                                          lr_size, sr_factor, base_frame_indices)
    print('Done!')

    return sr_imgs_adap_ker


def calcGST(inputIMG, w=3):
    img = inputIMG.astype(np.float32) / 255.0

    imgDiffX = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)
    imgDiffY = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)

    imgDiffX = cv.GaussianBlur(imgDiffX, (5, 5), 0)
    imgDiffY = cv.GaussianBlur(imgDiffY, (5, 5), 0)

    imgDiffXX = cv.multiply(imgDiffX, imgDiffX)
    imgDiffYY = cv.multiply(imgDiffY, imgDiffY)
    imgDiffXY = cv.multiply(imgDiffX, imgDiffY)

    J11 = cv.boxFilter(imgDiffXX, cv.CV_32F, (w, w))
    J22 = cv.boxFilter(imgDiffYY, cv.CV_32F, (w, w))
    J12 = cv.boxFilter(imgDiffXY, cv.CV_32F, (w, w))

    S = np.stack((J11, J12, J12, J22), axis=2).reshape(img.shape+(2, 2))
    S_ = np.stack((J22, -J12, -J12, J11), axis=2).reshape(img.shape+(2, 2))
    e_val, e_vec = np.linalg.eig(S_)

    eigenValues = np.zeros(img.shape + (2, 2))
    eigenVectors = np.zeros(img.shape + (2, 2))
    for i, _ in enumerate(e_val):
        for j, __ in enumerate(_):
            idx = e_val[i, j].argsort()[::-1]
            eigenValues[i, j] = np.diag(e_val[i, j][idx])
            eigenVectors[i, j] = e_vec[i, j][:, idx]

    coeh = cv.divide(eigenValues[:, :, 0, 0] - eigenValues[:, :, 1, 1],
                     eigenValues[:, :, 0, 0] + eigenValues[:, :, 1, 1])**2

    return S, eigenValues, eigenVectors, coeh, imgDiffX, imgDiffY


def cov_heuristic(eigenValues, eigenVectors, coeh, img_size,kDetail = .05):
    #kDetail = .33
    #kDetail = .05
    # kDetail = .15
    kDenoise = 4
    Dth = .005
    Dtr = .013
    kStretch = 4.0
    # kShrink = 2.0
    kShrink = .5

    L1 = eigenValues[:, :, 0, 0]
    A = 1 + np.sqrt(coeh)
    D = np.clip(1 - L1/Dtr + Dth, 0, 1)
    k1_ = kDetail * kStretch * A
    k2_ = kDetail / (kShrink * A)

    k1 = ((1-D)*k1_ + D*kDetail*kDenoise)**2
    k2 = ((1-D)*k2_ + D*kDetail*kDenoise)**2

    K = np.stack((k1, np.zeros(img_size), np.zeros(img_size), k2),
                 axis=2).reshape(img_size+(2, 2))

    sigma = np.zeros(img_size + (2, 2))
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            sigma[i, j] = eigenVectors[i, j].dot(K[i, j]).dot(np.transpose(eigenVectors[i, j]))

    return sigma


def compute_covariances(sr_imgs_rect_ker,kDetail = .05):
    print('\nComputing kernel covariance matrices...')
    _, eigenValues, eigenVectors, coeh, _, _ = calcGST(sr_imgs_rect_ker)
    cov_matrices = cov_heuristic(eigenValues, eigenVectors, coeh, sr_imgs_rect_ker.shape,kDetail = .05)
    print('Done!')

    return cov_matrices


def align_bgr_ecc(bgr_dict, template='red', number_of_iterations=15):
    bgr_dict_aligned = {}
    for ch in bgr_dict:
        if ch == template:
            bgr_dict_aligned[ch] = bgr_dict[ch]
        else:
            bgr_dict_aligned[ch], _ = align_ecc(bgr_dict[template], bgr_dict[ch],
                                                number_of_iterations=number_of_iterations)
    return bgr_dict_aligned


def save_bgr(bgr_dict_aligned, working_dir, file_name):
    cv.imwrite(working_dir + '/' + file_name, bgr_dict_aligned)


def main(args):
    print('Called with args:')
    print(vars(args))

    # Load LR frames
    input_path = args.input_path
    working_dir = input_path + '/../'
    LR_frames_dict, n_frames = load_LR_frames(working_dir, input_path, args.subcrop, nf=args.num_frames, sf=args.starting_frame)
    
    # Compute homographies and output aligned frames as a byproduct
    base_frame_indices = frame2index(LR_frames_dict, args.base_frame)
    print('base_index: ', base_frame_indices)
    warp_matrices, lr_size, = compute_homographies(LR_frames_dict, working_dir,
                                                   base_frame_indices, number_of_iterations=args.ecc_iter)

    # Transform base grids
    dewarp_grids = transform_base_grids(warp_matrices, lr_size, n_frames, working_dir, base_frame_indices)

    # Rectagular kernel reconstruction
    sr_factor = args.sr_factor
    z_string = str(float(sr_factor)).replace('.', '_')

    img_tables = build_img_tables(LR_frames_dict, lr_size, n_frames)
    sr_imgs_rect_ker = rect_ker_SR(dewarp_grids, img_tables, lr_size, sr_factor, base_frame_indices)
    save_bgr(sr_imgs_rect_ker, working_dir, 'SR_z' + z_string + '_fixed_kernel.png')

    # Adaptive kernel reconstruction
    cov_matrices = compute_covariances(sr_imgs_rect_ker,kDetail = args.kDetail)
    sr_imgs_adap_ker = adap_ker_SR(dewarp_grids, img_tables, cov_matrices, lr_size, sr_factor,
                                   base_frame_indices)
    save_bgr(sr_imgs_adap_ker, working_dir, 'SR_z' + z_string + '_adaptive_kernel.png')
