from collections import OrderedDict

import cv2
import numpy as np

################################################
# multiframe_sr / utils.py
# https://publicgitlab.satellogic.com/guillermo.becker/spotlight_sr/-/commit/c9eaae444365b771ad574c229595c52966ecb633
################################################


LOG_FORMAT = '%(asctime)s - %(module)s - %(levelname)s - %(message)s'


def filter_dict_by_key(dictionary, key):
    return OrderedDict({k: v for k, v in dictionary.items() if k != key})


def save_bgr(frames_per_band, output_dir, output_name, suffix=''):
    bgr_image = np.zeros(frames_per_band['blue'].shape + (3,), dtype=np.uint8)
    bgr_image[:, :, 0] = frames_per_band['blue']
    bgr_image[:, :, 1] = frames_per_band['green']
    bgr_image[:, :, 2] = frames_per_band['red']

    filename = f"{output_name}_RGB{suffix}.png"
    output_path = f"{output_dir}/{filename}"
    cv2.imwrite(output_path, bgr_image)


def sharp_kernel_5_by_5():
    return np.array(
        [[.0001, .001, .002, .001, .0001],
         [.001, .004, .006, .004, .001],
         [.002, .006, 1, .006, .002],
         [.001, .004, .006, .004, .001],
         [.0001, .001, .002, .001, .0001]]
    )


def zoom_x(image, zoom, interpolation=cv2.INTER_LINEAR):
    if zoom == 1:
        return image

    xsize = int(image.shape[1] * zoom)
    ysize = int(image.shape[0] * zoom)

    return cv2.resize(image, (xsize, ysize), interpolation=interpolation)

################################################
# multiframe_sr / alignment.py
# https://publicgitlab.satellogic.com/guillermo.becker/spotlight_sr/-/commit/c9eaae444365b771ad574c229595c52966ecb633
################################################

import cv2
import numpy as np


def ecc_alignment(template_image, input_image, motion_type=3, border=0, max_iter=15, eps=1e-10):
    """Aligns two images using the Enhanced Correlation Coefficient (ECC) criterion
    Uses find_ecc_homography function

    Args:
        template_image (np.array): single-channel image to use as template
        input_image (np.array): single-channel image that should be aligned
        motion_type (int, Optional): motion model to use. Default=3
        border (int, Optional): amount of rows and columns to remove
            from the borders before calling ECC. Default=0
        max_iter (int, Optional): The maximum number of iterations
            for the ECC criteria. Default=15
        eps (float, Optional): the desired accuracy at which the iterative algorithm stops

    Returns:
        A tuple (np.array, np.array) containing the aligned input image
            and the calculated warp matrix by ECC, respectively
    """

    template_image_xsize = template_image.shape[1]
    template_image_ysize = template_image.shape[0]

    if border > 0:
        template_image = template_image[border:-border - 1, border:-border - 1]
        input_image = input_image[border:-border - 1, border:-border - 1]

    warp_matrix = find_ecc_homography(
        template_image,
        input_image,
        motion_type=motion_type,
        max_iter=max_iter,
        eps=eps
    )

    if motion_type == cv2.MOTION_HOMOGRAPHY:  # Use warpPerspective for Homography
        aligned_input_image = cv2.warpPerspective(
            input_image,
            warp_matrix,
            (template_image_xsize, template_image_ysize),
            flags=cv2.INTER_LINEAR +
            cv2.WARP_INVERSE_MAP
        )
    else:  # Use warpAffine for Translation, Euclidean and Affine
        aligned_input_image = cv2.warpAffine(
            input_image,
            warp_matrix,
            (template_image_xsize, template_image_ysize),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return aligned_input_image, warp_matrix


def find_ecc_homography(
    template_image,
    input_image,
    motion_type=3,
    max_iter=15,
    eps=1e-10,
    pyramid_levels=1,
    pyramid_scale=1
):
    """Finds alignment homography using the Enhanced Correlation Coefficient (ECC) criterion.
    Allows scaling pyramid approach to boost performance

    Args:
        template_image (np.array): single-channel image to use as template
        input_image (np.array): single-channel image that should be aligned
        motion_type (int, Optional): motion model to use. Default=3
        max_iter (int, Optional): The maximum number of iterations
            for the ECC criteria. Default=15
        eps (float, Optional): the desired accuracy at which ECC stops.
            Default=1e-10
        pyramid_levels (int, Optional): number of levels for the pyramid.
            Default=1
        pyramid_scale (float, Optional): scale factor for the pyramid levels.
            Default=1. Ignored if levels=1

    Returns:
        np.array: warp matrix calculated by ECC

    motion_types allowed:
    0 - MOTION_TRANSLATION sets a translational motion model;
    1 - MOTION_EUCLIDEAN sets a Euclidean (rigid) transformation as motion model
    2 - MOTION_AFFINE sets an affine motion model (DEFAULT);
    3 - MOTION_HOMOGRAPHY sets a homography as a motion model

    For more information check OpenCV documentation
    https://docs.opencv.org/3.4/dc/d6b/group__video__track.html
    """
    criteria = define_ecc_criteria(max_iter, eps)

    warp_matrix, scale_matrix = init_ecc_matrices(motion_type, pyramid_scale)

    template_pyramid = construct_pyramid(template_image, pyramid_scale, pyramid_levels)
    input_pyramid = construct_pyramid(input_image, pyramid_scale, pyramid_levels)
    
    for level in range(pyramid_levels):
        try:
            cc, warp_matrix = cv2.findTransformECC(
                template_pyramid[level],
                input_pyramid[level],
                warp_matrix,
                motion_type,
                criteria
            )
        except:
            cc, warp_matrix = cv2.findTransformECC(
                template_pyramid[level],
                input_pyramid[level],
                warp_matrix,
                motion_type,
                criteria,
                None, 3
            )
        if level < pyramid_levels - 1:
            warp_matrix = warp_matrix * scale_matrix

    return warp_matrix


def construct_pyramid(image, scale_factor=2, levels=2):
    pyramid = [image]
    for level in range(levels - 1):
        new_level = cv2.resize(
            pyramid[0],
            None,
            fx=1/scale_factor,
            fy=1/scale_factor,
            interpolation=cv2.INTER_AREA
        )
        pyramid.insert(0, new_level)
    return pyramid


def homography_scale_matrix(s_factor):
    return np.array(
        [[1, 1, s_factor],
         [1, 1, s_factor],
         [1/s_factor, 1/s_factor, 1]],
        dtype=np.float32
    )


def affine_scale_matrix(s_factor):
    return np.array(
        [[1, 1, s_factor],
         [1, 1, s_factor]],
        dtype=np.float32
    )


def init_ecc_matrices(motion_type=cv2.MOTION_HOMOGRAPHY, s_factor=1):
    if motion_type == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        scale_matrix = homography_scale_matrix(s_factor)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        scale_matrix = affine_scale_matrix(s_factor)
    return warp_matrix, scale_matrix


def define_ecc_criteria(max_iter, eps):
    return (
        # The type of termination criteria
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max_iter,
        eps
    )


def compose_images_with_weights(images, weights, b=3):
    if b:
        slice_ = (slice(b, -b), slice(b, -b))
    else:
        slice_ = (slice(None, None), slice(None, None))

    weights_sum = np.zeros_like(weights[0][slice_])
    images_sum = np.zeros_like(weights[0][slice_])

    for i in range(len(images)):
        images_sum += images[i][slice_] * weights[i][slice_]
        weights_sum += weights[i][slice_]
    composition = images_sum / weights_sum
    return composition.astype(np.uint8), weights_sum


def apply_warp(im, warp_matrix, interpolation=cv2.INTER_LINEAR):
    return cv2.warpPerspective(im, warp_matrix, (im.shape[1], im.shape[0]), flags=interpolation + cv2.WARP_INVERSE_MAP)


################################################
# multiframe_sr / base.py
# https://publicgitlab.satellogic.com/guillermo.becker/spotlight_sr/-/commit/c9eaae444365b771ad574c229595c52966ecb633
################################################

import os
from abc import ABC, abstractmethod

class MultiFrameSR(ABC):
    """Base class for super resolution algorithms"""

    def __init__(self, crop_name, zoom=3, debug_folder=None):
        self._crop_name = crop_name
        self._zoom = zoom
        self._debug_folder = debug_folder
        if self._debug_folder:
            self._prepare_debug_folder(self._debug_folder)
            self._output_name = self._build_debug_output_name(crop_name, zoom)

    @abstractmethod
    def super_resolve(self, images_by_id, base_image_id):
        """Perfroms multi frame super resolution on base image

        Args:
            images_by_id (dict): a {str: np.array} dict with images to use
            base_image_id (str): key of images_by_id pointing to
                the base image to super resolve
        """
        raise NotImplementedError("Implement in subclass!")

    def _prepare_debug_folder(self, debug_folder):
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)

    def _build_debug_output_name(self, crop_name, zoom):
        z_string = str(float(zoom)).replace('.', '_')
        return f"{crop_name}_z{z_string}"

################################################
# multiframe_sr / warp_weights.py
# https://publicgitlab.satellogic.com/guillermo.becker/spotlight_sr/-/commit/c9eaae444365b771ad574c229595c52966ecb633
################################################

import numpy as np
import cv2
import pickle
import logging

#from multiframe_sr.base import MultiFrameSR
#from multiframe_sr.alignment import compose_images_with_weights, apply_warp
#from multiframe_sr.alignment import find_ecc_homography, homography_scale_matrix
#from multiframe_sr.utils import sharp_kernel_5_by_5, zoom_x

from collections import OrderedDict


logger = logging.getLogger(__name__)


class WarpWeights(MultiFrameSR):

    def __init__(self, crop_name, zoom=3, interpolation=cv2.INTER_LANCZOS4, debug_folder=None):
        self._interpolation = interpolation
        self._subsample, self._upsample = self._define_sampling(zoom)
        self._kernel = self._define_kernel(self._upsample)
        super().__init__(crop_name, zoom, debug_folder)

    def super_resolve(self, frames_by_id, base_frame_id):
        base_frame = frames_by_id[base_frame_id]
        aligned_frames_by_id = OrderedDict()
        aligned_frames_by_id[base_frame_id] = self._upsample_frame(base_frame)
        scale_matrix = homography_scale_matrix(self._upsample)
        base_weight = np.tile(self._kernel, base_frame.shape)
        weights = [base_weight]
        
        warp_matrices = {}
        for frame_id, frame in frames_by_id.items():
            if frame_id != base_frame_id:
                warp_matrix = find_ecc_homography(
                    base_frame,
                    frame,
                    eps=1e-3,
                    max_iter=15,
                    pyramid_levels=2,
                    pyramid_scale=2
                )
                upsampled_frame = self._upsample_frame(frame)
                warp_scaled_matrix = warp_matrix * scale_matrix
                aligned_frames_by_id[frame_id] = apply_warp(upsampled_frame, warp_scaled_matrix, interpolation=self._interpolation)
                weights.append(apply_warp(base_weight.copy(), warp_scaled_matrix))
                warp_matrices[frame_id] = warp_matrix

        if self._debug_folder:
            self._debug_aligned_frames(aligned_frames_by_id, base_frame_id)
            self._debug_warp_matrices(warp_matrices)

        aligned_frames = list(aligned_frames_by_id.values())
        composition, _ = compose_images_with_weights(aligned_frames, weights, b=0)
        return zoom_x(composition, 1 / self._subsample)

    def _define_kernel(self, upsample):
        if upsample == 3:
            return sharp_kernel_5_by_5()[1:-1, 1:-1]

        return sharp_kernel_5_by_5()

    def _define_sampling(self, zoom):
        if zoom == 1.5:
            upsample = 3  # warp_weights upsample on 3 or 5 factor only
            subsample = 2
        else:
            upsample = zoom
            subsample = 1

        return subsample, upsample

    def _upsample_frame(self, frame):
        return zoom_x(frame, self._upsample, interpolation=self._interpolation)

    def _debug_aligned_frames(self, aligned_frames, frame_base_id):
        base_frame = aligned_frames[frame_base_id]
        for frame_id, frame in aligned_frames.items():
            output_file = self._build_aligned_output_file(frame_id)
            cv2.imwrite(output_file, frame)
            diff = np.abs(base_frame.astype(np.int16)-frame)

            logger.debug(
                '{}: Aligned diff Mean:{}, Max:{},  Cropped Max:{}'
                .format(
                    frame_id,
                    diff.mean(),
                    diff.max(),
                    diff[10:-10, 10:-10].max()
                )
            )

    def _build_aligned_output_file(self, frame_id):
        return f'{self._debug_folder}/{self._output_name}_{frame_id}_ww_ecc_aligned.png'

    def _debug_warp_matrices(self, warp_matrices):
        output_file = f'{self._debug_folder}/{self._output_name}_ww_ecc_homographies.npy'
        with open(output_file, 'wb') as f:
            pickle.dump(warp_matrices, f, pickle.HIGHEST_PROTOCOL)


################################################
# multiframe_sr / factory.py
# https://publicgitlab.satellogic.com/guillermo.becker/spotlight_sr/-/commit/c9eaae444365b771ad574c229595c52966ecb633
################################################

import logging

# from multiframe_sr.fake import Fake
# from multiframe_sr.agk import AGK
# from multiframe_sr.warp_weights import WarpWeights

logger = logging.getLogger(__name__)

FAKE_SR = 'fake'
AGK_SR = 'agk'
WARPW_SR = 'warpw'


def create_super_resolver(kind, *args, **kwargs):
    classes = {
        # FAKE_SR: Fake,
        # AGK_SR: AGK,
        WARPW_SR: WarpWeights
    }

    if kind not in classes:
        raise NotImplementedError(
            f"Multi-Frame super resolution algorithm '{kind}' not implemented!"
        )

    return classes[kind](*args, **kwargs)


def sr_warp_weights(crops, im_base, zoom= 3):
    
    from tempfile import TemporaryDirectory

    sr_algo = "warpw"

    with TemporaryDirectory() as debug_folder:

        ww = create_super_resolver(sr_algo, crop_name='', zoom=zoom, interpolation=cv2.INTER_LANCZOS4, debug_folder=debug_folder)
        sr_crop = ww.super_resolve(crops, im_base)

    return sr_crop
