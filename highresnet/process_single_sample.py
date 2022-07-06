
import os
import json
import torch
import skimage
import numpy as np
from torch.utils.data import Dataset
from highresnet.hrnet import HRNet
from highresnet.data_loader import ImageSet

class collateFunction():
    """ Util class to create padded batches of data. """

    def __init__(self, min_L=32 ):
        """
        Args:
            min_L: int, pad length
        """
        self.min_L = min_L

    def __call__(self, batch):
        return self.collateFunction(batch)

    def collateFunction(self, batch):
        """
        Custom collate function to adjust a variable number of low-res images.
        Args:
            batch: list of imageset
        Returns:
            padded_lr_batch: tensor (B, min_L, W, H), low resolution images
            alpha_batch: tensor (B, min_L), low resolution indicator (0 if padded view, 1 otherwise)
            hr_batch: tensor (B, W, H), high resolution images
            hm_batch: tensor (B, W, H), high resolution status maps
            isn_batch: list of imageset names
        """
        
        lr_batch = []  # batch of low-resolution views
        alpha_batch = []  # batch of indicators (0 if padded view, 1 if genuine view)
        hr_batch = []  # batch of high-resolution views
        hm_batch = []  # batch of high-resolution status maps
        isn_batch = []  # batch of site names
        
        imageset = batch[0]
        lrs = imageset['lr']['lr']
        L, H, W = lrs.shape

        if L >= self.min_L:  # pad input to top_k
            lr_batch.append(lrs[:self.min_L])
            alpha_batch.append(torch.ones(self.min_L))
        else:
            pad = torch.zeros(self.min_L - L, H, W)
            lr_batch.append(torch.cat([lrs, pad], dim=0))
            alpha_batch.append(torch.cat([torch.ones(L), torch.zeros(self.min_L - L)], dim=0))

        padded_lr_batch = torch.stack(lr_batch, dim=0)
        alpha_batch = torch.stack(alpha_batch, dim=0)

        return padded_lr_batch, alpha_batch, hr_batch, hm_batch, isn_batch

class ImagesetDatasetSingle(Dataset):
    """
    Derived Dataset class for loading many imagesets from a list of directories.
    """
    
    def __init__(self, imageset):

        super().__init__()
        self.imageset = imageset
        
    def __len__(self):
        return 1     

    def __getitem__(self, index):
        """
        Returns an ImageSet dict of all assets in the directory of the given index.
        """

        imageset = self.imageset
        imset = [imageset]
        
        if len(imset) == 1:
            imset = imset[0]
        
        imset_list = imset if isinstance(imset, list) else [imset]
        for i, imset_ in enumerate(imset_list):
            imset_['lr'] = torch.from_numpy(skimage.img_as_float(imset_['lr']).astype(np.float32))
            if imset_['hr'] is not None:
                imset_['hr'] = torch.from_numpy(skimage.img_as_float(imset_['hr']).astype(np.float32))
                imset_['hr_map'] = torch.from_numpy(imset_['hr_map'].astype(np.float32))
            imset_list[i] = imset_
        
        if len(imset_list) == 1:
            imset = imset_list[0]
        
        return imset

def load_model(config, checkpoint_file):
    '''
    Loads a pretrained model from disk.
    Args:
        config: dict, configuration file
        checkpoint_file: str, checkpoint filename
    Returns:
        model: HRNet, a pytorch model
    '''
    # checkpoint_dir = config["paths"]["checkpoint_dir"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HRNet(config["network"]).to(device)
    model.load_state_dict(torch.load(checkpoint_file))
    return model

def get_sr(imset, model, min_L=16):
    '''
    Super resolves an imset with a given model.
    Args:
        imset: imageset
        model: HRNet, pytorch model
        min_L: int, pad length
    Returns:
        sr: tensor (1, C_out, W, H), super resolved image
    '''
    collator = collateFunction(min_L=min_L)
    lrs, alphas, hrs, hr_maps, names = collator([imset])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    
#     device='cpu'
#     model.to(device)
    
    
    
    lrs = lrs.float().to(device)
    alphas = alphas.float().to(device)
    
    sr = model(lrs, alphas)[:, 0]
    sr = sr.detach().cpu().numpy()[0]
    
    return sr

def img_rng(img):
    if len(img.shape)==2:
        return (img.min(), img.max())
    else:
        rng = []
        for b in range(3):
            rng.append((img[:,:,b].min(), img[:,:,b].max()))
        return rng

def img_set_rng(img, rng): # Output in float, make .astype(np.uint8) or type required
    old_rng = img_rng(img)
    sal = img.astype(np.float)
    if len(img.shape)==2:
        sal = ((sal-old_rng[0])/(old_rng[1]-old_rng[0]))*((rng[1]-rng[0]))+rng[0]
    else:
        for b in range(3):
            sal[:,:,b] = ((sal[:,:,b]-old_rng[b][0])/(old_rng[b][1]-old_rng[b][0]))*((rng[b][1]-rng[b][0]))+rng[b][0]
    return sal

def sr_high_res_net(crops, im_base, zoom, model,npdtype=np.uint16):
    
    lr_images = [crops[im_base]] + [crops[k] for k in crops if k!=im_base]
    
    lr_images = [ np.asarray(el, dtype=npdtype) for el in lr_images ]
    #lr_images = [ el.astype('uint16') for el in lr_images ]
    
    clearances = np.asarray([0 for el in lr_images])
    clearances[0] = 255
    
    # Organise all assets into an ImageSet (OrderedDict)
    imageset = ImageSet(name="whatever",
                        lr=np.array(lr_images),
                        hr=None,
                        hr_map=None,
                        clearances=clearances,
                        )
    
    MINL = 16
    
    imset = ImagesetDatasetSingle( imageset )
    
    # collator = collateFunction(min_L=MINL)
    # lrs, alphas, hrs, hr_maps, names = collator([imset])
    # np.save('/scratch/spotlight_pipeline/spotlight_sr/lrs.npy',lrs)
    #     print("sr = get_sr(imset,model,min_L=config['training']['min_L'])")
    #     print("np.save('/scratch/spotlight_pipeline/spotlight_sr/sr.npy',sr)")
    #     import pdb; pdb.set_trace()
    
    sr = get_sr( imset , model , min_L=MINL )
    sr = (255*sr).astype('uint8')
    #sr = skimage.img_as_ubyte( sr )
    #np.save('/scratch/spotlight_pipeline/spotlight_sr/sr.npy',sr)
    #sr = img_set_rng( sr , img_rng( crops[im_base] ) ).astype( np.uint8 )
    #sr = ( sr - sr.min() ) / ( sr.max() - sr.min() )
    #sr = skimage.img_as_uint( sr ).astype('uint8')
    return sr