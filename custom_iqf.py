import os
import tempfile
import sys
import json
import cv2
import piq
import torch
import numpy as np

from glob import glob
from typing import Any, Dict, Optional, List, Union
from iquaflow.datasets import DSModifier
from iquaflow.metrics import Metric
from iquaflow.experiments import ExperimentInfo

from swd import SlicedWassersteinDistance

from joblib import Parallel, delayed
from highresnet.process_single_sample import sr_high_res_net, load_model
from noai.sr_simple_bilardism_v0 import sr_bilardism
from noai.sr_simple_bilardism_v1 import sr_warp_weights
from noai.sr_adaptive_kernels_single_band import (
    load_LR_frames,
    frame2index,
    compute_homographies,
    transform_base_grids,
    build_img_tables,
    rect_ker_SR,
    compute_covariances,
    adap_ker_SR,
    mod_img_parallel
)

#########################
# SISR AI Methods
#########################

from msrn.msrn import load_msrn_model, inference_model

def process_file(
    nimg, model, compress=True, out_win=256,
    wind_size=512, stride=480, batch_size=1,
    scale=2, padding=5, manager=None
):
    
    H,W,C = nimg.shape
    
    nimg = nimg.astype(np.float)
    nimg /= 255
    
    nimg = cv2.copyMakeBorder(
        nimg,
        padding, padding, padding, padding,
        cv2.BORDER_REPLICATE
    )

    # inference
    result = inference_model(
        model, nimg,
        wind_size=wind_size, stride=stride,
        scale=scale, batch_size=batch_size,
        manager=manager, add_noise=None
    ) # you can add noise during inference to get smoother results (try from 0.1 to 0.3; the higher the smoother effect!) 

    result = result[2*padding:-2*padding,2*padding:-2*padding]
    result = cv2.convertScaleAbs(result, alpha=np.iinfo(np.uint8).max)
    result = result.astype(np.uint8)
    
    result = cv2.resize(result, (out_win,out_win), cv2.INTER_AREA)

    return result

    
def sisr_msrn( nimg , model,zoom=3,wind_size=128,gpu_device="0" ):
    
    res_output = 1/zoom # inria resolution
    
    result = process_file(
        nimg, model, compress=True, out_win=int(zoom*nimg.shape[-2]),
        wind_size=wind_size+10, stride=wind_size+10, scale=2,
        batch_size=1, padding=5
    )
    
    return result

def load_ai_model( 
    model_fn = "single_frame_sr/SISR_MSRN_X2_BICUBIC.pth",
    bucket_name="image-quality-framework",
    gpu_device = "0"
):
    # Rename to full bucket subdirs...
    model_s3_fn = os.path.join("iq-mfsr-use-case/models/weights",model_fn)
    
    print(model_fn)
    
    # Download files in a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        
        model_local_fn = os.path.join( tmpdirname , "model.pth" )
        
        for bucket_fn, local_fn in zip(
            [ model_s3_fn ],
            [ model_local_fn ]
        ):
            
            print( f"https://{bucket_name}.s3-eu-west-1.amazonaws.com/{bucket_fn}" )
            os.system( f"wget https://{bucket_name}.s3-eu-west-1.amazonaws.com/{bucket_fn} -O {local_fn}" )
        
        assert os.path.exists(model_local_fn), 'AWS model file not found'
        
        model = load_msrn_model(
                    weights_path=model_local_fn,
                    cuda=gpu_device
                )

    return model

#########################
# MFSR AI Methods
#########################

def load_ai_model_and_config( 
    config_fn = "hrn_exp27.json",
    model_fn = "exp27/HRNet_30.pth",
    bucket_name="image-quality-framework"
):
    # Rename to full bucket subdirs...
    config_s3_fn = os.path.join(
        "iq-mfsr-use-case/models/config",
        config_fn.replace('+',r'%2B')
    )
    model_s3_fn = os.path.join(
        "iq-mfsr-use-case/models/weights",
        model_fn.replace('+',r'%2B')
    )
    
    print(model_fn)
    
    # Download files in a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        
        config_local_fn = os.path.join( tmpdirname , "config.json" )
        model_local_fn = os.path.join( tmpdirname , "HRNet.pth" )
        
        for bucket_fn, local_fn in zip(
            [ config_s3_fn , model_s3_fn ],
            [ config_local_fn , model_local_fn ]
        ):
            print( f"https://{bucket_name}.s3-eu-west-1.amazonaws.com/{bucket_fn}" )
            os.system( f"wget https://{bucket_name}.s3-eu-west-1.amazonaws.com/{bucket_fn} -O {local_fn}" )
        
        assert os.path.exists(config_local_fn), 'AWS config file not found'
        assert os.path.exists(model_local_fn), 'AWS model file not found'
        
        with open( config_local_fn , "r" ) as read_file:
            config = json.load( read_file )
        
        model = load_model( config , model_local_fn )
        
    config["training"]["create_patches"] = False

    return model

#########################
# MFSR no AI Methods
#########################

def agk(input_path,zoom,lr_name_filter='LR*.png'):
    
    class Args():
        def __init__(self):
            pass
    
    args = Args()
    
    kDetail = (.05 if zoom==3 else 0.15)
    print("zoom , kDetail = ", zoom,kDetail)
    
    setattr( args, 'input_path', input_path )
    _ = [setattr( args, name, 25 ) for name in ['ei','ecc_iter']]
    _ = [setattr( args, name, None ) for name in ['nf','num_frames']]
    _ = [setattr( args, name, None ) for name in ['sf','starting_frame']]
    _ = [setattr( args, name, None ) for name in ['bf','base_frame']]
    _ = [setattr( args, name, [0, 0, 0, 0] ) for name in ['sc','subcrop']]
    _ = [setattr( args, name, float(zoom) ) for name in ['sr','sr_factor']]
    _ = setattr( args, 'kDetail', float(kDetail) )
    
    # Load LR frames
    input_path = args.input_path
    working_dir = input_path + '/../'
    LR_frames_dict, n_frames = load_LR_frames(
        working_dir,
        input_path,
        args.subcrop,
        lr_name_filter = lr_name_filter,
        nf=args.num_frames,
        sf=args.starting_frame
    )
    
    if len(LR_frames_dict[[k for k in LR_frames_dict][0]].shape)==3:
        LR_frames_dict = { k:LR_frames_dict[k][...,0] for k in LR_frames_dict }
    
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
    #save_bgr(sr_imgs_rect_ker, working_dir, 'SR_z' + z_string + '_fixed_kernel.png')

    # Adaptive kernel reconstruction
    cov_matrices = compute_covariances(sr_imgs_rect_ker, kDetail = args.kDetail)
    sr_imgs_adap_ker = adap_ker_SR(dewarp_grids, img_tables, cov_matrices, lr_size, sr_factor,
                                   base_frame_indices)
    #save_bgr(sr_imgs_adap_ker, working_dir, 'SR_z' + z_string + '_adaptive_kernel.png')
    
    return sr_imgs_adap_ker

def zoom_x(img, z, interpolation=cv2.INTER_LINEAR):
    return img if z==1 else cv2.resize(img, (int(img.shape[1]*z), int(img.shape[0]*z)), interpolation=interpolation)

#########################
# Modifiers
#########################

class DSModifierMFSR(DSModifier):
    """
    Class derived from DSModifier that modifies a dataset iterating its folder.

    Args:
        ds_modifer: DSModifier. Composed modifier child

    Attributes:
        name: str. Name of the modifier
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier
    """
    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "algo":"fake",
            "zoom": 3,
            "n_jobs":1,
            "config": "hrn_exp27.json",
            "model": "exp27/HRNet_30.pth"
        },
    ):
        
        if not 'n_jobs' in params:
            params['n_jobs'] = 1
        
        assert all([key in params for key in ['algo','zoom','n_jobs']]), \
            "Some required keys are missing in the params dict"
        
        if params['n_jobs']>1 and params['algo']!='agk':
            print(f"n_jobs > 1 Not supported yet for algorithm {params['algo']},")
            print("changing it to params['n_jobs']=1 ...")
            params['n_jobs']=1
        
        algo = params['algo']
        
        if algo=='hrn':
            subname = algo + '_' + os.path.splitext(params['config'])[0]+'_'+os.path.splitext(params['model'])[0].replace('/','-')
            self.name = f"mfsr+{subname}_modifier"
        else:
            self.name = f"mfsr+{algo}_modifier"
        
        self.params: Dict[str, Any] = params
        self.ds_modifier = ds_modifier
        self.params.update({"modifier": "{}".format(self._get_name())})
        
        if self.params["algo"]=="hrn":
            
            # Load HighResNet weights
            
            self.model = load_ai_model_and_config(
                config_fn = ("hrn_exp27.json" if "config" not in self.params else self.params["config"]),
                model_fn = ("exp27/HRNet_30.pth" if "model" not in self.params else self.params["model"] )
            )
            
        elif self.params["algo"]=="msrn":
            
            self.model = load_ai_model(
                model_fn = ("single_frame_sr/SISR_MSRN_X2_BICUBIC.pth" if "model" not in self.params else self.params["model"]),
                gpu_device = "0"
            )
        else:
            
            self.model = None

    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        """Modify images
        Iterates the data_input path loading images, processing with _mod_img(), and saving to mod_path

        Args
            data_input: str. Path of the original folder containing images
            mod_path: str. Path to the new dataset
        Returns:
            Name of the new folder containign the images
        """
        input_name = os.path.basename(data_input)
        dst = os.path.join(mod_path, input_name)
        os.makedirs(dst, exist_ok=True)
        
        print(f'For each subdir in <{data_input}>...')
        
        if self.params['n_jobs']>1:
            
            from noai.sr_adaptive_kernels_single_band import mod_img_parallel
            
            _ = Parallel(n_jobs=self.params['n_jobs'],verbose=100)(
                delayed(mod_img_parallel)([dst,data_input,imgset_subdir,self])
                for imgset_subdir in os.listdir( data_input )
            )
        
        else:
            
            for imgset_subdir in os.listdir( data_input ):

                lr_fn_lst = glob( os.path.join(data_input,imgset_subdir,'LR*.png') )

                if not len(lr_fn_lst):
                    continue

                try:
                    imgp = self._mod_img( lr_fn_lst )
                    cv2.imwrite( os.path.join(dst, imgset_subdir+'+'+self.params['algo']+'.png'), imgp )
                except Exception as e:
                    print(e)
            
        return input_name

    def _mod_img(self, lr_fn_lst: List[str]) -> np.array:
        
        zoom = self.params["zoom"]
        cut = 4*zoom//2
        cutw = 1
        
        crops =  {
            str(int( os.path.basename(lrfn).replace('LR','').replace('.png','') )):cv2.imread(lrfn)
            for lrfn in lr_fn_lst
        }
        
        if len(crops['0'].shape)==3:
            crops = { k:crops[k][...,0] for k in crops }
        
        if self.params["algo"]=="fake":
            
            rec_img = zoom_x(crops['0'], zoom)[cut:-cut, cut:-cut]
            
        elif self.params["algo"]=="warpw_v0":
            
            iml = [crops['0']]
            for im in crops:
                if im != '0':
                    iml.append(crops[im])
            rec_img = sr_bilardism(iml, zoom=zoom)[cutw:-cutw, cutw:-cutw]

        elif self.params["algo"]=="warpw":
            
            rec_img = sr_warp_weights(crops,'0', zoom=zoom)[cut:-cut, cut:-cut]
        
        elif self.params["algo"]=="agk":
            
            rec_img = agk( os.path.dirname(lr_fn_lst[0]) , zoom, lr_name_filter='LR*.png' )[cut:-cut, cut:-cut]
        
        elif self.params["algo"]=="hrn":
            
            rec_img = sr_high_res_net(crops, '0', zoom, self.model,npdtype=np.uint8)[cut:-cut, cut:-cut]
        
        elif self.params["algo"]=="msrn":
            
            rec_img = sisr_msrn(
                np.stack([crops['0'],crops['0'],crops['0']],axis=-1) ,
                self.model,
                zoom=zoom,
                wind_size=128,
                gpu_device="0"
            )[cut:-cut, cut:-cut,0]
        
        else:
            
            raise "Algo not detected"
            
        print(crops['0'].shape,rec_img.shape)
        
        return rec_img

#########################
# Similarity Metrics
#########################

class SimilarityMetricsForMFSR( Metric ):
    
    def __init__( self, experiment_info: ExperimentInfo, cut: Optional[int] = 6 , n_jobs: Optional[int] = 1 ) -> None:
        self.metric_names = ['ssim','psnr','gmsd','mdsi','haarpsi','fid']
        self.cut = cut
        self.n_jobs = n_jobs
        self.experiment_info = experiment_info
    
    def _parallel(self, fid:object, pred_fn:str) -> List[Dict[str,Any]]:

        cut = self.cut
        # pred_fn be like: xview_id1529imgset0012+hrn.png
        subdir = os.path.basename(pred_fn).split('+')[0]
        gt_fn = glob( os.path.join(self.data_path,"L0",f"{subdir}","HR.png") )[0]
        
        pred = cv2.imread( pred_fn )/255
        gt = cv2.imread( gt_fn )/255
        
        pred = ( pred[...,0] if len(pred.shape)==3 else pred )
        gt = ( gt[...,0] if len(gt.shape)==3 else gt )
        
        gt = gt[cut:-cut,cut:-cut]
        
        pred = torch.from_numpy( pred )
        gt = torch.from_numpy( gt )
        
        pred = pred.view(1,-1,pred.shape[-2],pred.shape[-1])
        gt = gt.view(1,-1,gt.shape[-2],gt.shape[-1])
        
        results_dict = {
            "ssim":piq.ssim(pred,gt).item(),
            "psnr":piq.psnr(pred,gt).item(),
            "gmsd":piq.gmsd(pred,gt).item(),
            "mdsi":piq.mdsi(pred,gt).item(),
            "haarpsi":piq.haarpsi(pred,gt).item(),
            "fid":fid( torch.squeeze(pred), torch.squeeze(gt) ).item()
        }

        return results_dict

    def apply(self, predictions: str, gt_path: str) -> Any:
        """
        In this case gt_path will be a glob criteria to the HR images
        """
        
        # These are actually attributes from ds_wrapper
        self.data_path = os.path.dirname(gt_path)
        self.parent_folder = os.path.dirname(self.data_path)
        
        # predictions be like /mlruns/1/6f1b6d86e42d402aa96665e63c44ef91/artifacts'
        guessed_run_id = predictions.split(os.sep)[-3]
        modifier_subfold = [
            k
            for k in self.experiment_info.runs
            if self.experiment_info.runs[k]['run_id']==guessed_run_id
        ][0]
        
        pred_fn_lst = glob(os.path.join( self.parent_folder, modifier_subfold,'L0','*.png' ))
        stats = { met:0.0 for met in self.metric_names }
        cut = self.cut
        
        fid = piq.FID()

        results_dict_lst = Parallel(n_jobs=self.n_jobs,verbose=30)(
            delayed(self._parallel)(fid,pred_fn)
            for pred_fn in pred_fn_lst
            )
        
        stats = {
            met:np.median([
                r[met]
                for r in results_dict_lst
                if 'float' in type(r[met]).__name__
                ])
            for met in self.metric_names
        }
                
        return stats

class SlicedWassersteinMetric( Metric ):
    
    def __init__(
        self,
        experiment_info: ExperimentInfo,
        cut: Optional[int] = 6,
        n_jobs: Optional[int] = 1,
        ext: str = 'png',
        n_pyramids:Union[int, None]=None,
        slice_size:int=7,
        n_descriptors:int=128,
        n_repeat_projection:int=128,
        proj_per_repeat:int=4,
        device:str='cpu',
        return_by_resolution:bool=False,
        pyramid_batchsize:int=128
    ) -> None:
        
        self.cut                  = cut
        self.n_jobs               = n_jobs
        self.ext                  = ext
        self.metric_names         = ['swd']
        self.experiment_info      = experiment_info
        self.n_pyramids           = n_pyramids
        self.slice_size           = slice_size
        self.n_descriptors        = n_descriptors
        self.n_repeat_projection  = n_repeat_projection
        self.proj_per_repeat      = proj_per_repeat
        self.device               = device
        self.return_by_resolution = return_by_resolution
        self.pyramid_batchsize    = pyramid_batchsize

    def _parallel(self, swdobj:object, pred_fn:str) -> List[Dict[str,Any]]:

        cut = self.cut
        # pred_fn be like: xview_id1529imgset0012+hrn.png
        subdir = os.path.basename(pred_fn).split('+')[0]
        gt_fn = glob( os.path.join(self.data_path,"L0",f"{subdir}","HR.png") )[0]
        
        pred = cv2.imread( pred_fn )/255
        gt = cv2.imread( gt_fn )/255
        
        pred = ( pred[...,0] if len(pred.shape)==3 else pred )
        gt = ( gt[...,0] if len(gt.shape)==3 else gt )
        
        gt = gt[cut:-cut,cut:-cut]
        
        pred = np.stack([pred,pred,pred],axis=0)
        gt = np.stack([gt,gt,gt],axis=0)
        
        pred = torch.from_numpy( pred )
        gt = torch.from_numpy( gt )
        
        pred = pred.view(1,*pred.shape)
        gt = gt.view(1,*gt.shape)

        results_dict = {'swd':swdobj.run(pred.double(),gt.double()).item()}

        return results_dict
    
    def apply(self, predictions: str, gt_path: str) -> Any:
        """
        In this case gt_path will be a glob criteria to the HR images
        """
        
        # These are actually attributes from ds_wrapper
        self.data_path = os.path.dirname(gt_path)
        self.parent_folder = os.path.dirname(self.data_path)
        
        # predictions be like /mlruns/1/6f1b6d86e42d402aa96665e63c44ef91/artifacts'
        guessed_run_id = predictions.split(os.sep)[-3]
        modifier_subfold = [
            k
            for k in self.experiment_info.runs
            if self.experiment_info.runs[k]['run_id']==guessed_run_id
        ][0]
        
        pred_fn_lst = glob(os.path.join( self.parent_folder, modifier_subfold,'L0','*.png' ))
        stats = { met:0.0 for met in self.metric_names }
        cut = self.cut
        
        swdobj = SlicedWassersteinDistance(
            n_pyramids           = self.n_pyramids,
            slice_size           = self.slice_size,
            n_descriptors        = self.n_descriptors,
            n_repeat_projection  = self.n_repeat_projection,
            proj_per_repeat      = self.proj_per_repeat,
            device               = self.device,
            return_by_resolution = self.return_by_resolution,
            pyramid_batchsize    = self.pyramid_batchsize
        )

        results_dict_lst = Parallel(n_jobs=self.n_jobs,verbose=30)(
            delayed(self._parallel)(swdobj,pred_fn)
            for pred_fn in pred_fn_lst
            )
        
        stats = {
            met:np.median([
                r[met]
                for r in results_dict_lst
                if 'float' in type(r[met]).__name__
                ])
            for met in self.metric_names
        }
                
        return stats