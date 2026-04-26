import torch
import os
from easydict import EasyDict as edict
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import random
from PIL import Image
from source.rendering.transform_perspective import compose_rotmat
from source.rendering.pano2perspective import GetPerspective
from source.sky_histogram import compute_sky_histogram


class vigor_dataset(torch.utils.data.Dataset):
    def __init__(self, 
                 render_size = 256,
                 sr_factor = 2,
                 root = None,
                 is_train = True,
                 data_txt = None,
                 input_resize = None,
                    ):
        if root is None:
            raise ValueError("`root` must point to the prepared VIGOR dataset directory.")
        if data_txt is None:
            raise ValueError("`data_txt` must point to a VIGOR split file.")
        self.sr_factor = sr_factor
        self.render_size = render_size
        self.root = root
        self.is_train = is_train
        self.input_resize = input_resize
        if self.input_resize is not None:
            self.sat_view_resize =  T.Compose([         
                                            T.Resize((input_resize, input_resize),interpolation=InterpolationMode.BICUBIC),  
                                            T.ToTensor(),                    
                                            T.Normalize(                      
                                            mean=[0.485, 0.456, 0.406],                
                                            std=[0.229, 0.224, 0.225]              
                                            )])
        self.data_txt = data_txt
        print('data_txt:', self.data_txt)

        with open(self.data_txt, "r") as f:
            self.data = f.readlines()


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx].strip().split(" ")
        if len(data) == 5:
            sat_path, str_path, sky_mask_path, h, w = data
        else:
            sat_path, str_path, sky_mask_path = data
            h,w = 0,0
        #  order: h, w ,z
        position = np.array([float(h)/ 320.,float(w)/ 320.,-0.85])
        # all the position should be in the range of -1 to 1
        assert position.min() >= -1 and position.max() <= 1

        sat_img = Image.open(os.path.join(self.root, sat_path)).convert('RGB')
        str_img = Image.open(os.path.join(self.root, str_path)).convert('RGB')
        sky_mask = Image.open(os.path.join(self.root, sky_mask_path)).convert('L')
        
        if self.input_resize is not None:
            sat_img_input = self.sat_view_resize(sat_img)

        if self.is_train:
            Roll = 0
            #random in [-30,30]
            Pitch = random.uniform(-20,20)
            Yaw = random.uniform(-179,179)
            RollPitchYaw = torch.tensor([Roll,Pitch,Yaw]).float()
            fovx_list = [90,105,120]
            fovy_list = [90,105,120]
            fovx = random.choice(fovx_list)
            fovy = random.choice(fovy_list)
        else:
            RollPitchYaw = torch.tensor([0,0,0]).float()
            fovx = 120
            fovy = 120
        R_c2w = compose_rotmat(RollPitchYaw[0], RollPitchYaw[1], RollPitchYaw[2])
        per_h_w  = self.render_size
        fx = 0.5 * per_h_w  / np.tan(0.5 * fovx / 180.0 * np.pi) # w
        fy = 0.5 * per_h_w  / np.tan(0.5 * fovy / 180.0 * np.pi) # h
        cx = (per_h_w - 1) / 2.0
        cy = (per_h_w - 1) / 2.0
        intrinsics = [fx, fy, cx, cy]
        intrinsics_save = np.array([[intrinsics[0]/2, 0, intrinsics[2]/2],
                                [0, intrinsics[1]/2, intrinsics[3]/2],
                                [0, 0, 1]])
        [pespective,per_sky_mask] = GetPerspective([np.array(str_img),np.array(sky_mask)], intrinsics,RollPitchYaw[2].item() ,RollPitchYaw[1].item(), per_h_w, per_h_w)
        pespective = torch.from_numpy(pespective.transpose(2, 0, 1)).float().div(255).mul(2).add(-1)
        per_sky_mask = torch.from_numpy(per_sky_mask).float().div(255).unsqueeze(0).round()
        per_sky_img = (pespective.add(1).mul(0.5) * per_sky_mask).mul(2).add(-1).float() # data range [-1, 1]

        # Image.fromarray(pespective).save('pespective.png')

        # resize (sr_factor is always 2)
        sat_img = sat_img.resize((self.render_size, self.render_size))
        str_img = str_img.resize((self.render_size*2, self.render_size//2))
        sky_mask = sky_mask.resize((self.render_size*2, self.render_size//2), Image.NEAREST)



        # to tensor
        sat_img = torch.from_numpy(np.array(sat_img).transpose(2, 0, 1)).float().div(255).mul(2).add(-1)
        str_img = torch.from_numpy(np.array(str_img).transpose(2, 0, 1)).float().div(255).mul(2).add(-1)
        str_img_flag = torch.tensor([1]).float()
        sky_mask = torch.from_numpy(np.array(sky_mask)).float().div(255).unsqueeze(0)


        sky_image = (str_img.add(1).mul(0.5) * sky_mask).mul(2).add(-1).float() # data range [-1, 1]
        sky_histo = torch.from_numpy(
            compute_sky_histogram(sky_image.numpy(), hist_range=(-1, 1))
        ).float()

        out_dict = edict()
        out_dict.sat_img = sat_img
        out_dict.sat_path = sat_path
        out_dict.illu_fea = sky_histo
        out_dict.str_path = str_path
        satellite_dir_name = 'satellite/'
        if os.path.exists(os.path.join(self.root, sat_path.replace(satellite_dir_name, 'sat_depth/').replace('.png','_depth.png'))):
            sat_dep  = Image.open(os.path.join(self.root, sat_path.replace(satellite_dir_name, 'sat_depth/').replace('.png','_depth.png'))).convert('RGB')
            sat_dep = sat_dep.resize((self.render_size, self.render_size)).convert('L')
            sat_dep  = torch.from_numpy(np.array(sat_dep)).float().div(255).unsqueeze(0)
            out_dict.sat_dep = sat_dep
        if self.input_resize is not None:
            out_dict.sat_img_input = sat_img_input

        out_dict.str_img = str_img
        out_dict.str_img_flag = str_img_flag
        out_dict.sky_mask = sky_mask
        out_dict.sky_image = sky_image
        # for pespective rendering
        position[0]  *= - 1 # -h w z
        position = position[[1,0,2]]  # w -h z  change to the same coordinate system with opensfm east X, North Y, Up Z.
        c2w = np.eye(4)
        c2w[:3, :3] = np.array(R_c2w)
        c2w[:3, 3] = position
        out_dict.c2w = torch.from_numpy(c2w).float()
        out_dict.intrinsics = torch.tensor(intrinsics_save).float()
        out_dict.RollPitchYaw = RollPitchYaw
        out_dict.per_img = pespective
        out_dict.per_sky_mask = per_sky_mask
        out_dict.per_sky_img = per_sky_img
        out_dict.image_sat_name = sat_path
        return out_dict
