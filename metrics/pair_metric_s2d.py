# python3.8
"""Contains the class to evaluate EG3D model with
Frechet Inception Distance (FID) .

FID metric is introduced in paper https://arxiv.org/pdf/1706.08500.pdf
"""

import os
import shutil
import numpy as np
from pathlib import Path

import torch

import lpips
from pytorch_msssim import ssim
import torchvision
import cv2
from torchvision.transforms import ToPILImage
to_pil = ToPILImage()
from PIL import ImageDraw
from PIL import Image
from cleanfid import fid

__all__ = ['pair_metric_sat2density']


# save torch tensor with pillow, input range is [0,1]
def save_image(image, path):
    # clamp
    image = image.clamp(0, 1)
    assert image.min() >= 0 and image.max() <= 1, 'Image should be in range [0,1]'
    if len(image.shape) == 4:
        image = image.squeeze(0)
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)


class pair_metric_sat2density():
    """Defines the class for FID metric computation."""

    def __init__(self,
                 work_dir=None,
                 save_img=True,
                 save_all=False,
                 cat_img2save = False,
                 two_view_input_mode=True,
                 data_path=None,
                 sky_path=None,
                 sky_from_training=False,
                 save_triplane=False,
                 save_sat=False,
                 save_pano=True,
                 save_per=False,
                 temp_save=False,
                 score_test=True,
                 save_DSM=False
                 ):
        """Initializes the class with number of real/fakes samples for FID.

        Args:
            real_num: Number of real images used for FID evaluation. If not set,
                all images from the given evaluation dataset will be used.
                (default: -1)
        """
        self.work_dir = work_dir
        self.temp_save = temp_save
        if save_img:
            if self.temp_save:
                self.work_dir = os.path.join(self.work_dir, 'temp')
                if os.path.exists(self.work_dir):
                    shutil.rmtree(self.work_dir, ignore_errors=True)
                    print(f"Removed temporary directory: {self.work_dir}")

        self.save_img = save_img
        self.two_view_input_mode = two_view_input_mode
        self.save_all = save_all
        self.save_triplane = save_triplane
        self.cat_img2save = cat_img2save
        self.score_test = score_test
        if self.cat_img2save:
            self.score_test = False

        self.save_sat = save_sat
        self.save_pano = save_pano
        self.save_per = save_per
        if self.cat_img2save or self.save_all:
            self.save_sat = True

        self.save_DSM = save_DSM
        if self.save_DSM:
            self.save_sat = True
            self.save_pano = False
            self.save_per = False
            self.score_test = False
            self.save_triplane = False
            sky_path = None
            sky_from_training = False

        if self.score_test:
            self.loss_fn_alex = lpips.LPIPS(net='alex',eval_mode=True).cuda()
            self.loss_fn_sque = lpips.LPIPS(net='squeeze',eval_mode=True).cuda()

        self.sky_from_training = sky_from_training



        if self.sky_from_training:
            self.histo_numpy = np.load(os.path.join(data_path, 'histo.npy'))
            
        self.img_save_path = os.path.join(self.work_dir, 'images')
        if self.save_all:
            self.img_save_path = os.path.join(self.work_dir, 'images_all')
        else:
            if self.save_img == True:
                self.img_save_path = os.path.join(self.work_dir, 'images')
            if self.cat_img2save:
                self.img_save_path = os.path.join(self.work_dir, 'cat_img2save')
        if sky_path:
            self.sky_path = sky_path
            if data_path and (data_path not in self.sky_path):
                self.sky_path = os.path.join(data_path, self.sky_path)
                assert os.path.isfile(self.sky_path),self.sky_path
            self.frozen_histo_sky = make_histo(self.sky_path)
            sky_path_base_name = os.path.basename(self.sky_path).rsplit('.',1)[0]
            if  self.save_all or self.cat_img2save or self.save_img:
                self.img_save_path = self.img_save_path+sky_path_base_name
        else:
            self.frozen_histo_sky = None

        

        if self.sky_from_training:
            self.img_save_path = self.img_save_path + '_training_shuffle'
        if  self.save_all or self.cat_img2save or self.save_img:
            os.makedirs(self.img_save_path,exist_ok=True)
            print(f'img_save_path is {self.img_save_path}')




            

        # make sure  self.sky_path is a file 
    def extract_pair_metric(self, data_loader, model,accelerator=None):
        """Extracts inception features from fake data."""
        accelerator.print('Validating...')
        model.eval()


        all_features = []

        if not os.path.exists(self.img_save_path):
            os.makedirs(self.img_save_path, exist_ok=True)
        for idx, batch_data in enumerate(data_loader):
            real_img = batch_data['render_str_img'] if  'render_str_img' in batch_data.keys() else batch_data['str_img']


            if self.two_view_input_mode:
                with torch.no_grad():
                    batch_result = model(batch_data['sat_img_input'],
                                    batch_data['str_img_input'],
                                    syn_sat  = self.save_sat, 
                                    syn_pano = self.save_pano,
                                    syn_per  = self.save_per,
                                    intrinsics=batch_data['intrinsics'],
                                    c2w = batch_data['input_c2w'] if 'input_c2w' in batch_data.keys() else batch_data['c2w'],
                                    coordinates=None,
                                    render_c2w = batch_data['render_c2w'] if 'render_c2w' in batch_data.keys() else batch_data['c2w'],
                                    )
                batch_result_image = batch_result.str_output.image_raw_compo.mul(2).add(-1)
                
            else:

                # panorama
                input_sat = batch_data['sat_img_input'] if 'sat_img_input' in batch_data.keys() else batch_data['sat_img']

                # input_img = batch_data['image_sat'].detach()

                # mask_sky = batch_data['mask_sky']
                # mask_opa = ((1-mask_sky) > 0.5).repeat(1,3,1,1)
                # real_img = batch_data['image_grd'].detach()
                # # real sky region
                # real_sky = real_img.clone()
                # real_sky[mask_opa] =-1
                if self.sky_from_training:
                    bs = batch_data['sat_img_input'].size(0)
                    num_histos = self.histo_numpy.shape[0]
                    random_idx = np.random.randint(num_histos, size=bs)
                    sky_histos = torch.from_numpy(self.histo_numpy[random_idx]).float().cuda()
                else:
                    if self.frozen_histo_sky is None:
                        sky_histos = batch_data['illu_fea'].cuda().detach()
                    else:
                        sky_histos = self.frozen_histo_sky.repeat(input_sat.size(0), 1)



                if 'position' in batch_data.keys():
                    position = batch_data['position']
                else:
                    position = None
                with torch.no_grad():
                    batch_result = model(input_sat,
                                            z_ill=sky_histos,
                                            syn_sat=self.save_sat,
                                            syn_pano=self.save_pano,   
                                            syn_per=self.save_per, 
                                            intrinsics=batch_data['intrinsics'],
                                            c2w=batch_data['c2w'],
                                            coordinates=None,
                                            random_sat_crop=False,
                                            )
                    if 'str_output' in batch_result.keys():
                        batch_result_image = batch_result.str_output.sr_image.mul(2).add(-1) if 'sr_image' in batch_result.str_output.keys() else batch_result.str_output.image_raw_compo.mul(2).add(-1)



            if self.score_test:
                batch_psnr = 10 * torch.log10(4 / (batch_result_image - real_img).pow(2).mean(dim=(1, 2, 3))).unsqueeze(-1)
                batch_ssim = ssim(batch_result_image/2.+0.5, real_img/2.+0.5, data_range=1., size_average=False).unsqueeze(-1)

                # compute lpips between batch_result_image and real_img, data range is [-1,1]
                batch_lpips_alex = self.loss_fn_alex(batch_result_image, real_img).squeeze(-1).squeeze(-1)
                batch_lpips_sque = self.loss_fn_sque(batch_result_image, real_img).squeeze(-1).squeeze(-1)
                gathered_psnr = accelerator.gather_for_metrics(batch_psnr.detach()) if accelerator is not None else batch_psnr

                gathered_ssim = accelerator.gather_for_metrics(batch_ssim.detach()) if accelerator is not None else batch_ssim
                gathered_lpips_alex = accelerator.gather_for_metrics(batch_lpips_alex.detach()) if accelerator is not None else batch_lpips_alex
                gathered_lpips_sque = accelerator.gather_for_metrics(batch_lpips_sque.detach()) if accelerator is not None else batch_lpips_sque
                if accelerator.is_main_process:
                    gathered_all_result = np.concatenate([gathered_psnr.detach().cpu().numpy(), gathered_ssim.detach().cpu().numpy(), gathered_lpips_alex.detach().cpu().numpy(), gathered_lpips_sque.detach().cpu().numpy()], axis=1)
                    
                if accelerator.is_main_process:
                    # if type all_features is list
                    if type(all_features) == list:
                    # if all_features == []:
                        all_features = gathered_all_result
                    else:
                        all_features = np.concatenate([all_features, gathered_all_result], axis=0)
                if idx % 20 == 0:
                    accelerator.print('Validating:',idx,'/',str(len(data_loader)),'psnr:',gathered_psnr.detach().cpu().numpy().mean(),'lpips_alex:',gathered_lpips_alex.detach().cpu().numpy().mean(),'lpips_sque:',gathered_lpips_sque.detach().cpu().numpy().mean(),'ssim:',gathered_ssim.detach().cpu().numpy().mean())
            else:
                if idx % 20 == 0:
                    accelerator.print('Validating:',idx,'/',str(len(data_loader)))

            if self.save_triplane:
                for i,image_sat_path in enumerate(batch_data['image_sat_name']):
                    save_path = os.path.join(self.work_dir, 'triplane')
                    # mkdir if not exist
                    os.makedirs(save_path, exist_ok=True)
                    print(os.path.join(save_path,os.path.basename(image_sat_path)))
                    # cv2.imwrite(os.path.join(save_path,os.path.basename(image_sat_path)), batch_result['triplanes'][i].cpu().numpy())
                    # save to npy file
                    np.save(os.path.join(save_path,os.path.basename(image_sat_path).replace('.png','.npy')), batch_result['triplanes'][i].cpu().numpy())

            if self.cat_img2save:
                input_img = batch_data['sat_img'].cpu()
                real_img = real_img.cpu()
                batch_result_image = batch_result_image.cpu()
                gray_image = batch_result.str_output.image_depth.cpu()
                for i in range(len(batch_data['str_path'])):
                    # input img size is 3,256,256
                    # real img size is 3,128,512
                    # fake img size is 3,128,512
                    # cat together to save, size is 3,256, 768
                    if 'c2w' in batch_data.keys():
                        # position_i = batch_data['position'][i]
                        position_i = batch_data.c2w[i][:2, 3].cpu().numpy()
                    else:
                        position_i = [0,0]
                    input_img_i = input_img[i]*0.5+0.5
                    size = input_img_i.size(1)
                    input_img_i = to_pil(input_img_i)
                    draw = ImageDraw.Draw(input_img_i)
                    x,y,radius = position_i[0]*size/2 + size/2, (-position_i[1])*size/2 + size/2, 3
                    # x is the width, y is the height, so position[1] is width, position[0] is height
                    # # temp ignore
                    # draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill='red')
                    input_img_i = torchvision.transforms.ToTensor()(input_img_i)

                    cat_img = torch.cat([batch_result_image[i]*0.5+0.5,real_img[i]*0.5+0.5],dim=1)
                    cat_img_numpy = cat_img.mul(255).cpu().numpy()
                    cat_img_numpy = np.transpose(cat_img_numpy, (1, 2, 0))
                    cat_img_numpy = cv2.cvtColor(cat_img_numpy, cv2.COLOR_BGR2RGB)
                    
                    gray_image_i = gray_image[i] / gray_image[i].max()
                    gray_image_i = gray_image_i.clamp(0, 1)*255
                    gray_image_i = gray_image_i.squeeze().cpu().numpy().astype(np.uint8)
                    result_street_dep = cv2.applyColorMap(gray_image_i, cv2.COLORMAP_PLASMA)
                    opacity = batch_result.str_output.alpha_raw[i].squeeze().cpu().numpy() < 0.5
                    result_street_dep[opacity] = [0, 0, 0]

                    shape = result_street_dep.shape
                    result_street_dep = cv2.resize(result_street_dep, (shape[1]*2, shape[0]*2), interpolation=cv2.INTER_LINEAR)

                    result_street = np.concatenate([cat_img_numpy, result_street_dep], axis=0)

                    #resize input_img_i to [h of result_street, h of result_street]
                    input_img_i = input_img_i.mul(255).cpu().numpy()
                    input_img_i = np.transpose(input_img_i, (1, 2, 0))
                    input_img_i = cv2.cvtColor(input_img_i, cv2.COLOR_BGR2RGB)
                    h_street = result_street.shape[0]
                    input_img_i = cv2.resize(input_img_i, (h_street, h_street), interpolation=cv2.INTER_LINEAR)

                    cat_img = np.concatenate([input_img_i, result_street], axis=1)
                    
                    # save sat depth
                    result_sat_dep =  batch_result.sat_output.image_depth[i]
                    result_sat_dep = 1- (result_sat_dep- result_sat_dep.min())/(2-result_sat_dep.min())
                    result_sat_dep = result_sat_dep.clamp(0, 1)*255
                    result_sat_dep = result_sat_dep.squeeze().cpu().numpy().astype(np.uint8)
                    result_sat_DSM_MAP = cv2.applyColorMap(result_sat_dep, cv2.COLORMAP_VIRIDIS) # map
                    result_sat_dep     = cv2.applyColorMap(result_sat_dep, cv2.COLORMAP_PLASMA)

                    # resize to 448 * 448
                    result_sat_dep = cv2.resize(result_sat_dep, (448,448), interpolation=cv2.INTER_LINEAR)

                    # save ref depth
                    image_sat_dep = batch_data['sat_dep'][i]
                    image_sat_dep = (image_sat_dep- image_sat_dep.min())/(image_sat_dep.max()-image_sat_dep.min())
                    image_sat_dep = image_sat_dep.clamp(0, 1)*255
                    image_sat_dep = image_sat_dep.squeeze().cpu().numpy().astype(np.uint8)
                    image_sat_dep = cv2.applyColorMap(image_sat_dep, cv2.COLORMAP_PLASMA)
                    # resize to 448 * 448
                    image_sat_dep = cv2.resize(image_sat_dep, (448,448), interpolation=cv2.INTER_LINEAR)
                    # cat, shape is 448, 896
                    cat_dep = np.concatenate([image_sat_dep, result_sat_dep], axis=1)

                    cat_img = np.concatenate([cat_dep,cat_img], axis=0)


                    # only for generate DSM
                    input_img_i = cv2.resize(input_img_i, (256,256), interpolation=cv2.INTER_LINEAR)
                    cat_DSM_img = np.concatenate([input_img_i,result_sat_DSM_MAP], axis=1)
                    # COLORMAP_JET

                    save_type = 'cat'
                    # if .jpg change to .png
                    save_name = os.path.basename(batch_data['sat_path'][i]).rsplit('.',1)[0]+f'.png'
                    save_name = os.path.join(self.img_save_path, save_name)
                    # mkdir if not exist
                    os.makedirs(os.path.dirname(save_name), exist_ok=True)

                    # # temp ignore for generate sat cat DSM
                    # cv2.imwrite(save_name, cat_img)

                    cv2.imwrite(save_name, cat_DSM_img)

            elif self.save_DSM:
                DSM_SAVE_PATH = os.path.join(self.work_dir, 'DSM_result')
                DSM_VISUAL_PATH = os.path.join(self.work_dir, 'DSM_visual')
                if accelerator.is_main_process:
                    os.makedirs(DSM_SAVE_PATH, exist_ok=True)
                    os.makedirs(DSM_VISUAL_PATH, exist_ok=True)


                for i in range(len(batch_data['image_sat_name'])):
                    result_sat_dep =  batch_result.sat_output.image_depth[i]
                    result_sat_dep = 2 - result_sat_dep
                    # from [0,2] to [0,1]
                    result_sat_dep = result_sat_dep/2
                    # if "satellite_47" in batch_data['image_sat_name'][i]:
                    #     zoom_level = 20
                    #     pixel_number = 640
                    #     # ta/vigor/Seattle_DSM/satellite_47.56847531848441_-122.29516701917808_dsm.npz
                    #     lat = float(os.path.basename(batch_data['image_sat_name'][i]).split('_')[1]) 
                    #     spatial_data = calculate_spatial_info(lat,zoom_level,pixel_number)
                    #     spatial_resolution = spatial_data['size_m']
                    # else:
                    #     raise ValueError('city not found')
                    # result_sat_dep_meter = result_sat_dep * spatial_resolution
                    # convert to numpy and then save as npz file
                    np.savez(os.path.join(DSM_SAVE_PATH, os.path.basename(batch_data['image_sat_name'][i]).rsplit('.',1)[0]+'.npz'), result_sat_dep.squeeze().cpu().numpy())

                    # # save sat depth
                    # result_sat_dep =  batch_result.sat_output.image_depth[i]
                    # result_sat_dep = 1- (result_sat_dep- result_sat_dep.min())/(2-result_sat_dep.min())
                    # result_sat_dep = result_sat_dep.clamp(0, 1)*255
                    # result_sat_dep = result_sat_dep.squeeze().cpu().numpy().astype(np.uint8)
                    # result_sat_DSM_MAP = cv2.applyColorMap(result_sat_dep, cv2.COLORMAP_VIRIDIS) # map
                    # cv2.imwrite(os.path.join(DSM_VISUAL_PATH, os.path.basename(batch_data['image_sat_name'][i])), result_sat_DSM_MAP)
                    # print(f"Saved visualization images to ", os.path.join(DSM_VISUAL_PATH, os.path.basename(batch_data['image_sat_name'][i])))
                    

            elif self.save_img:
                str_render_name = 'render_str_path' if 'render_str_path' in batch_data.keys() else 'str_path'
                for i in range(len(batch_data[str_render_name])):
                    if not self.save_all:
                        save_name = os.path.basename(batch_data[str_render_name][i]).rsplit('.',1)[0]+'.png'
                        img = batch_result_image[i].mul(0.5).add(0.5)
                        save_image(img, os.path.join(self.img_save_path, save_name))


                        work_dir_path = Path(self.work_dir)
                        if not self.temp_save:
                            real_img_save_path = str(work_dir_path.parent.parent / 'real_img')
                        else:
                            real_img_save_path = str(work_dir_path / 'real_img')
                        if not os.path.exists(real_img_save_path):
                            os.makedirs(real_img_save_path, exist_ok=True)
                        save_name_ = os.path.join(real_img_save_path, save_name)
                        # save_input_name = os.path.join(input_save_path, save_name)s
                        if not os.path.exists(save_name_):
                            real_img_ = real_img[i].mul(0.5).add(0.5)
                            save_image(real_img_, save_name_)

                            
                    if self.save_all:
                        self.save_img2_path(batch_result_image, batch_data['str_path'])
                        self.save_img2_path([input_img,batch_result['sat_dep']], batch_data['str_path'],save_type='cat_sat_result')
                        
                        self.save_img2_path(image_depth, batch_data['str_path'],save_type='depth')
                        if self.frozen_histo_sky is None:
                            self.save_img2_path(batch_result['sky_img'], batch_data['str_path'], save_type='sky')
                        # save alpha raw
                        self.save_img2_path(batch_result['alpha_raw'], batch_data['str_path'], save_type='alpha_raw')
                        # save input_img
                        self.save_img2_path(input_img, batch_data['str_path'], save_type='input_img')
                        # save image_raw
                        self.save_img2_path(batch_result['image_raw'], batch_data['str_path'], save_type='image_raw')
                        # save sat_dep
                        self.save_img2_path(batch_result['sat_dep'], batch_data['str_path'], save_type='sat_dep')
                        # save sat_img_raw
                        self.save_img2_path(batch_result['sat_img_raw'], batch_data['str_path'], save_type='sat_img_raw')
                        # real sky
                        self.save_img2_path(real_sky, batch_data['str_path'], save_type='real_sky')





        accelerator.wait_for_everyone()
        if self.save_img and  accelerator.is_main_process and self.score_test:
            fid_score = fid.compute_fid(self.img_save_path, real_img_save_path,batch_size=512,num_workers=0,use_dataparallel=False)
            accelerator.print(f'fid_score: {fid_score:.4f}')
            # kid
            kid_score = fid.compute_kid(self.img_save_path, real_img_save_path,batch_size=512,num_workers=0,use_dataparallel=False)
            accelerator.print(f'kid_score: {kid_score:.4f}')

            if self.temp_save:
                shutil.rmtree(self.img_save_path, ignore_errors=True)
                accelerator.print(f"Removed temporary directory: {self.img_save_path}")
                shutil.rmtree(real_img_save_path, ignore_errors=True)
                accelerator.print(f"Removed temporary directory: {real_img_save_path}")

            print(f'fid:{fid_score:.4f},kid_score:{kid_score:.4f}')
        else:
            all_features = None
            fid_score = None
            kid_score = None
        accelerator.wait_for_everyone()
        return all_features, fid_score, kid_score
        
        

    def save_img2_path(self, img, name, save_type=''):
        
        for i in range(len(name)):
            # save_dir_per_img = os.path.dirname(os.path.join(self.img_save_path, os.path.dirname(name[i])))

            os.makedirs(self.img_save_path, exist_ok=True)
            # if '.jpg' change to '.png'
            if name[i].endswith('.jpg'):
                name[i] = name[i].replace('.jpg', '.png')
            save_name = os.path.join(self.img_save_path, os.path.basename(name[i]).rsplit('.',1)[0])
            os.makedirs(save_name, exist_ok=True)
            save_name = os.path.join(save_name, 'street.png' if save_type == '' else f'{save_type}.png')
            # mkdir if not exist
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            if save_type == 'sky':
                img = img.cpu()
                # from 'streetview/t1DOCdyniuWDC5JPqm4MWA_grdView.png' to 'streetview/t1DOCdyniuWDC5JPqm4MWA_grdView_{save_type}.png'
                # path: self.img_save_path + name[i] + '_' + save_type[i] + '.png'
                torchvision.utils.save_image(img[i]*0.5+0.5, save_name)
            if save_type == '':
                img = img.cpu()
                torchvision.utils.save_image(img[i]*0.5+0.5, save_name)
            if save_type == 'real_sky':
                img = img.cpu()
                torchvision.utils.save_image(img[i]*0.5+0.5, save_name)
            if save_type == 'depth':
                image = img[i].cpu().numpy()
                depth_max = image.max() 
                # where image > 4, set to 4
                image[image>depth_max] = depth_max
                depth = np.array((image/depth_max)*255,dtype=np.uint8)
                # to numpy image format
                depth = np.transpose(depth, (1, 2, 0))
                depth = cv2.applyColorMap(depth, cv2.COLORMAP_PLASMA)
                cv2.imwrite(save_name, depth)
                # torchvision.utils.save_image(img[i], os.path.join(self.img_save_path, f'{name[i]}_depth.png'))
            if save_type == 'cat_sat_result':
                input_img = img[0][i].cpu().clamp(-1,1).add(1).mul(0.5).mul(255).numpy().astype(np.uint8)
                result_sat_dep   = img[1][i].cpu()
                # to numpy image format
                input_img = np.transpose(input_img, (1, 2, 0))
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                result_sat_dep = 1- (result_sat_dep- result_sat_dep.min())/(result_sat_dep.max()-result_sat_dep.min())
                result_sat_dep = result_sat_dep.clamp(0, 1)*255
                result_sat_dep = result_sat_dep.squeeze().cpu().numpy().astype(np.uint8)
                result_sat_dep = cv2.applyColorMap(result_sat_dep, cv2.COLORMAP_PLASMA)
                # cat
                cat_img = np.concatenate([input_img, result_sat_dep], axis=1)
                cv2.imwrite(save_name, cat_img)
            if save_type == 'alpha_raw':
                # one chanel grey image
                opa = img[i].cpu().numpy().transpose(1,2,0)
                opa = opa*255
                opa = opa.astype(np.uint8)
                cv2.imwrite(save_name, opa)
            if save_type == 'input_img':
                input_img = img[i].cpu().add(1).mul(0.5)
                save_image(input_img,save_name)

            # save image_raw
            if save_type == 'image_raw':
                image_raw = img[i].cpu().clamp(-1,1).add(1).mul(0.5)
                save_image(image_raw,save_name)
            if save_type == 'sat_dep':
                image = img[i].cpu().numpy()
                depth_max = image.max() 
                # where image > 4, set to 4
                # image[image>depth_max] = depth_max
                depth = np.array((1- image/depth_max)*255,dtype=np.uint8)
                # to numpy image format
                depth = np.transpose(depth, (1, 2, 0))
                depth = cv2.applyColorMap(depth, cv2.COLORMAP_PLASMA)
                cv2.imwrite(save_name, depth)
            if save_type == 'sat_img_raw':
                image_raw = img[i].cpu().clamp(-1,1).add(1).mul(0.5)
                save_image(image_raw,save_name)


    def evaluate(self, data_loader, generator,accelerator):
        all_result,fid,kid = self.extract_pair_metric(data_loader, generator,accelerator)
        if accelerator.is_main_process and self.score_test:
            final_result = []
            num_score = len(all_result[1])
            for i in range(num_score):
                final_result.append(all_result[:,i].mean())
            name_list = ['metric_psnr','metric_ssim','metric_lpips_alex','metric_lpips_sque']
            result = {}
            for i,name in enumerate(name_list):
                result[name] =  float(final_result[i])

            if fid is not None:
                result['metric_fid'] = fid
            if kid is not None:
                result['metric_kid'] = kid
            msg_all = ''
            for key in result:
                prefix = f'{key}: '
                msg = f'{prefix}{result[key]:.3f}, '
                msg_all +=  msg 
            msg_all = 'Evaluating: ' + msg_all 
            accelerator.print(msg_all)
            
        else:
        #     assert all_result is None
            result = None
        return result


    def save(self, result, accelerator=None,global_step=0):
        if not accelerator.is_main_process:
            assert result is None
            return

        assert isinstance(result, dict)
        accelerator.log(result, step=global_step)





def make_histo(str_path,to_tensor=True,return_sky_img=False):
    grd_img_path = str_path
    grd_img = Image.open(grd_img_path).convert('RGB').resize((512,128))
    grd_img = np.array(grd_img)
    grd_img = torch.from_numpy(grd_img).permute(2, 0, 1).unsqueeze(0).float()
    # to [-1,1]
    grd_img = grd_img / 255.0 * 2.0 - 1.0

    if 'streetview/panos' in grd_img_path:
        mask_img_path = grd_img_path.replace('streetview/panos','sky_mask')
    else:
        if 'streetview' in grd_img_path:
            mask_img_path = grd_img_path.replace('streetview','pano_sky_mask')      
        elif   'panorama' in grd_img_path:
            mask_img_path = grd_img_path.replace('panorama','pano_sky_mask')
    mask_img_path = mask_img_path.replace('jpg','png') if mask_img_path.endswith('.jpg') else mask_img_path
    mask_img = Image.open(mask_img_path).convert('L').resize((512,128))
    mask_img = np.array(mask_img)
    mask_img = torch.from_numpy(mask_img).unsqueeze(0).unsqueeze(0).float()/ 255.0
    # round to 0 or 1
    mask_img = (mask_img > 0.5).float()

    sky_image = (grd_img+1.)*mask_img
    sky_image = sky_image.detach().numpy()
    from source.sky_histogram import compute_sky_histogram
    sky_histo = compute_sky_histogram(sky_image[0], hist_range=(0, 2))
    if return_sky_img:
        if to_tensor:
            return torch.from_numpy(sky_histo).unsqueeze(0).float().cuda(), sky_image[0]*0.5
        else:
            return sky_histo, sky_image[0]
    else:
        if to_tensor:
            return torch.from_numpy(sky_histo).unsqueeze(0).float().cuda()
        else:
            return sky_histo
