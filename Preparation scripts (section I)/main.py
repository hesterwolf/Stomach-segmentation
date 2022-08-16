# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 16:12:01 2022

@author: s132525
"""

import generateJSON
import random_split_patients
import XML_to_Nifti
import os

#fill in your path and task_name
task_name = 'Task002_Stomach'
your_path = r'D:\nnUNet_models'

#if you want to train a 2D model, set 'train2D = True' 
train2D = False

#if you want to check if the binary mask correctly fits the MRI, set 'check_masks = True'
check_masks = False


root = os.path.join(your_path, task_name)
if not os.path.exists(root):
    os.mkdir(root)
root1 = os.path.join(root, 'raw_data')
if not os.path.exists(root1):
    os.mkdir(root1)

if __name__ == '__main__':
    if len(os.listdir(root1))>0:
        XML_to_Nifti.XML_to_3DNifti(root, check_masks)
        if train2D == True:
            XML_to_Nifti.Nifti2D(root, check_masks)
            dim = '2D'
        else:
            dim = '3D'
        random_split_patients.split_patients(root, dim)
        generateJSON.jsonfile(root, task_name)
    else:
        print('The raw_data folder is created. Fill this folder with your patient data as described in the README.md and then run this script again.')
    
    