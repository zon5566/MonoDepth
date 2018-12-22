#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import os
import sys
import csv
import config
import argparse
import matplotlib.pyplot as plt
import matplotlib as mat
mat.use('Agg')


# In[ ]:


params, _ = config.get_config()
width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351

def load_input_images():
    filepath_arr = np.genfromtxt(os.path.join('utils', 'kitti_stereo_2015_test_files.txt'), dtype='U30', delimiter=' ')
    
    left_images = []
    for line in filepath_arr:
        filename = os.path.join('evaluation/ground_truth', line[0])[:-4] + '.png'
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        left_images.append(img)
    
    return left_images
    
def load_gt_disp(path):
    gt_disp = []
    for i in range(200):
        disp = cv2.imread(os.path.join(path, 'training/disp_noc_0', str(i).zfill(6)+'_10.png'), -1)
        disp = disp.astype(np.float32) / 255
        gt_disp.append(disp)
        
    return gt_disp
            
def convert_disp_to_depth(gt_disparities, pred_disparities):
    gt_depths = []
    pred_depths = []
    pred_disparities_resized = []
    
    for i in range(len(gt_disparities)):
        gt_disp = gt_disparities[i]
        height, width = gt_disp.shape

        pred_disp = pred_disparities[i]
        pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR)

        pred_disparities_resized.append(pred_disp) 

        mask = gt_disp > 0

        gt_depth = width_to_focal[width] * 0.54 / (gt_disp + (1.0 - mask))
        pred_depth = width_to_focal[width] * 0.54 / pred_disp

        gt_depths.append(gt_depth)
        pred_depths.append(pred_depth)
    return gt_depths, pred_depths, pred_disparities_resized

def compute_errors(gt, pred):
    thresh = np.maximum(gt/pred, pred/gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()
    
    rmse = (gt-pred)**2
    rmse = np.sqrt(rmse.mean())
    
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

if __name__ == '__main__':
    # an argument <model_type> is needed in command
    if not params.pp:
        pred_disp_filename = '{}/npy/disparities_{}.npy'.format(params.output_directory, sys.argv[1])
    else:
        pred_disp_filename = '{}/npy/disparities_{}_pp.npy'.format(params.output_directory, sys.argv[1])
    
    gt_disp_filename = os.path.join(params.output_directory, params.gt_directory)
    
    num_samples = 200

    pred_disps = np.load(pred_disp_filename)
    gt_disps = load_gt_disp(gt_disp_filename)
    
    gt_depths, pred_depths, pred_disps_resized = convert_disp_to_depth(gt_disps, pred_disps)
    
    # save input and predicted depth images
    fig_directory_path = 'evaluation/img/{}/'.format(sys.argv[1])
    input_images = load_input_images()
    for i in range(num_samples):
        if params.pp:
            filename = fig_directory_path + 'pred_pp_{:03d}.png'.format(i)
        else:
            filename = fig_directory_path + 'pred_{:03d}.png'.format(i)
            
        plt.figure()
        #plt.subplot(2,1,1)
        plt.imshow(input_images[i], cmap=plt.get_cmap('plasma'))
        plt.axis('off')
        plt.savefig('{}input_{}.png'.format(fig_directory_path, i), bbox_inches='tight', pad_inches=0)
        plt.close()
        
        #plt.subplot(2,1,2)
        plt.figure()
        plt.imshow(pred_disps_resized[i], cmap=plt.get_cmap('plasma'))
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        
        plt.close()
    ###
    
    rms     = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    d1_all  = np.zeros(num_samples, np.float32)
    a1      = np.zeros(num_samples, np.float32)
    a2      = np.zeros(num_samples, np.float32)
    a3      = np.zeros(num_samples, np.float32)
    
    if not params.pp:
        csv_filename = os.path.join(params.output_directory, 'csv', 'errors_{}.csv'.format(sys.argv[1]))
    else:
        csv_filename = os.path.join(params.output_directory, 'csv', 'errors_{}_pp.csv'.format(sys.argv[1]))
        
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['#Image', 'Abs Rel', 'Sq Rel', 'RMSE', 'RMSE log', 'D1-all', '<1.25', '<1.25^2', '<1.25^3']
        writer = csv.DictWriter(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(num_samples):
            gt_depth = gt_depths[i]
            pred_depth = pred_depths[i]

            pred_depth[pred_depth < params.min_depth] = params.min_depth
            pred_depth[pred_depth > params.max_depth] = params.max_depth

            gt_disp = gt_disps[i]
            mask = gt_disp > 0
            pred_disp = pred_disps_resized[i]

            disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
            bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff/gt_disp[mask]) >= 0.05)
            d1_all[i] = (bad_pixels.sum() / mask.sum()) * 100

            abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])
            
            writer.writerow({
                '#Image': i,
                'Abs Rel': abs_rel[i],
                'Sq Rel': sq_rel[i],
                'RMSE': rms[i],
                'RMSE log': log_rms[i],
                'D1-all': d1_all[i],
                '<1.25': a1[i],
                '<1.25^2': a2[i],
                '<1.25^3': a3[i]
            })
        
        writer.writerow({
                '#Image': 'average',
                'Abs Rel': np.mean(abs_rel),
                'Sq Rel': np.mean(sq_rel),
                'RMSE': np.mean(rms),
                'RMSE log': np.mean(log_rms),
                'D1-all': np.mean(d1_all),
                '<1.25': np.mean(a1),
                '<1.25^2': np.mean(a2),
                '<1.25^3': np.mean(a3)
        })

