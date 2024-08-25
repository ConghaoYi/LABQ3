import numpy as np
import matplotlib.pyplot as plt
import os
import skimage
import scipy
import pickle
import functools
from PIL import Image

import LABQ3
script_dir = os.path.dirname(os.path.realpath(__file__))

# IF AVAILABLE _USE CACHED CLASSIFIERS. YOU MUST REBUILD IF YOU CHANGE ANY OF THE STANDARDS,
# NOT NEEDED IF YOU JUST CHANGE THE INPUT IMAGES
USE_CACHED_BAYES = True

def get_config_binary(std1="CaCO3", std2="CdCO3", energy="8kev", step=1):
    unique_run_key = f"{std1}__{std2}__{energy}__{step}"
    config = dict(# Below related to the classifier
                  std1_name=std1,
                  std2_name=std2,
                  dir_std1=f"{script_dir}/standards/{std1}_{energy}/txt",
                  dir_std2=f"{script_dir}/standards/{std2}_{energy}/txt",
                  dir_out=f"{script_dir}/runs/",
                  step=step,
                  unique_run_key=unique_run_key,
                  use_pre_existing=USE_CACHED_BAYES,
                  # Below relates to the image
                  scan_name='bi_example',
                  dir_scan=f"{script_dir}/examples/data/binary",
                  down_sample_scale=False,
                  erosion=False,
                  median_filter=False,
                  scale_erosion=False,
                  scale_median=False,
                  )
    return config
        
def get_config_ternary(std1="CaCO3", std2="CdCO3", std3="ZnCO3", step=1):
    unique_run_key = f"{std1}__{std2}__{std3}__{step}"
    config = dict(# Below relates to the bayes classifier
                  std1_name=std1,
                  std2_name=std2,
                  std3_name=std3,
                  dir_std1_8kev=f"{script_dir}/standards/{std1}_8kev/txt",
                  dir_std2_8kev=f"{script_dir}/standards/{std2}_8kev/txt",
                  dir_std3_8kev=f"{script_dir}/standards/{std3}_8kev/txt",
                  dir_std1_10kev=f"{script_dir}/standards/{std1}_10kev/txt",
                  dir_std2_10kev=f"{script_dir}/standards/{std2}_10kev/txt",
                  dir_std3_10kev=f"{script_dir}/standards/{std3}_10kev/txt",
                  dir_out=f"{script_dir}/runs/",
                  step=step,
                  use_pre_existing=USE_CACHED_BAYES,
                  unique_run_key=unique_run_key,
                  # Below relates to the image
                  scan_name='ter_example',                  
                  dir_scan8=f"{script_dir}/examples/data/ternary/8kev",
                  dir_scan10=f"{script_dir}/examples/data/ternary/10kev",
                  down_sample_scale=False,
                  erosion=False,
                  median_filter=False,
                  scale_erosion=False,
                  scale_median=False,
                  )
    return config

LABQ3.main_bi(**get_config_binary("CdCO3", "CaCO3","8kev"))
LABQ3.main_ter(**get_config_ternary())
####################################
##          Example plots         ##
####################################

##Binary
def recreate_bi(f_std1,f_std2,out_dir):
    print("Generating an example RGB image for the binary particle\n")
    comp1 = np.load(f_std1)
    comp2 = np.load(f_std2)
    red_scale = comp1 * 255
    green_scale = comp2 * 255
    rgb_array = np.zeros((comp1.shape[0],comp1.shape[1],3))

    for i in range(rgb_array.shape[0]):
        for j in range(rgb_array.shape[1]):
            if np.isnan(red_scale[i][j]) or np.isnan(green_scale[i][j]):
                rgb_array[i][j] =(0,0,0)
            else:
                rgb_array[i][j] =(int(red_scale[i][j]),int(green_scale[i][j]),0)
    rgbimage = Image.fromarray(rgb_array.astype('uint8'),mode="RGB")
    rgbimage.save(out_dir)

recreate_bi(
    os.path.join(f'{script_dir}','runs/CdCO3__CaCO3__8kev__1/bi_example/std2/0001.npy'),
    os.path.join(f'{script_dir}','runs/CdCO3__CaCO3__8kev__1/bi_example/std1/0001.npy'),
    os.path.join(f'{script_dir}','runs/CdCO3__CaCO3__8kev__1/bi_example/my_slice.png')
)

##Ternary
def recreate_ter(f_std1,f_std2,f_std3,out_dir):
    print("Generating an example RGB image for the ternary particle\n")
    comp1 = np.load(f_std1)
    comp2 = np.load(f_std2)
    comp3 = np.load(f_std3)
    red_scale = comp1 * 255
    green_scale = comp2 * 255
    blue_scale = comp3*255
    rgb_array = np.zeros((comp1.shape[0],comp1.shape[1],3))
    for i in range(rgb_array.shape[0]):
        for j in range(rgb_array.shape[1]):
            if np.isnan(red_scale[i][j]) or np.isnan(green_scale[i][j]) or np.isnan(blue_scale[i][j]):
                rgb_array[i][j] =(0,0,0)
            else:
                rgb_array[i][j] =(int(red_scale[i][j]),int(green_scale[i][j]),int(blue_scale[i][j]))
    rgbimage = Image.fromarray(rgb_array.astype('uint8'),mode="RGB")
    rgbimage.save(out_dir)

recreate_ter(
    os.path.join(f'{script_dir}','runs/CaCO3__CdCO3__ZnCO3__1/ter_example/std1/0001.npy'),
    os.path.join(f'{script_dir}','runs/CaCO3__CdCO3__ZnCO3__1/ter_example/std2/0001.npy'),
    os.path.join(f'{script_dir}','runs/CaCO3__CdCO3__ZnCO3__1/ter_example/std3/0001.npy'),
    os.path.join(f'{script_dir}','runs/CaCO3__CdCO3__ZnCO3__1/ter_example/my_slice.png')
)

