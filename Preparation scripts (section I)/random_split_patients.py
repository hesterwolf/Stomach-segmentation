# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 09:30:11 2021

@author: s132525
"""
from collections import OrderedDict
import os
import random
import numpy as np
import json
import shutil
import pickle
import nibabel as nib

#Random split of training (85%) and test (15%)

def split_patients(root, dim): 
    root1 = os.path.join(root, 'raw_data')
    root2 = os.path.join(root, 'model_data')
    studies = os.listdir(root1)
    ListPatients= []
    for study in studies:
        ListPatients = ListPatients + os.listdir(os.path.join(root1, study, 'MRI'))
    random.shuffle(ListPatients)
    #
    
    LengthTrain = int(len(ListPatients)*.85)
    LengthTest = len(ListPatients)-LengthTrain
    
    print('Number of patients in trainingset:', LengthTrain)
    print('Number of patients in testset:', LengthTest)
    
    ListTrain = ListPatients[:LengthTrain]
    ListTest = ListPatients[LengthTrain:]
    
    #Save as text files
    with open(os.path.join(root2,'patientsTrain.txt'), 'w') as f:
        f.write(json.dumps(ListTrain))
    with open(os.path.join(root2, 'patientsTest.txt'), 'w') as f:
        f.write(json.dumps(ListTest))
    
    #Create splits_final.pkl file and put into imagesTr, labelsTr and imagesTs
    if not os.path.exists(os.path.join(root, 'Input voor nnUNet')):
        os.mkdir(os.path.join(root, 'Input voor nnUNet'))
    if not os.path.exists(os.path.join(root, 'Input voor nnUNet', 'imagesTr')):
        os.mkdir(os.path.join(root, 'Input voor nnUNet', 'imagesTr'))
    if not os.path.exists(os.path.join(root, 'Input voor nnUNet', 'labelsTr')):
        os.mkdir(os.path.join(root, 'Input voor nnUNet', 'labelsTr'))
    if not os.path.exists(os.path.join(root, 'Input voor nnUNet', 'imagesTs')):
        os.mkdir(os.path.join(root, 'Input voor nnUNet', 'imagesTs'))
        
        
    splits = []
    splits.append(OrderedDict())
    
    ListSlicesTrain = []
    ListSlicesTest = [] 
       
    if dim == '3D':
        root3 = os.path.join(root2, '3D', 'all_images')
        for study in os.listdir(root3):
            for patient in ListTrain:
                if patient in os.listdir(os.path.join(root3, study)):
                    path2MRI = os.path.join(root3, study, patient)
                    MRI_volumes = os.listdir(path2MRI)
                    for volume in MRI_volumes:
                        file_name = patient + '_' + volume[:-7]
                        ListSlicesTrain.append(file_name)  
                        shutil.copyfile(os.path.join(path2MRI, volume), os.path.join(root, 'Input voor nnUNet', 'imagesTr', file_name+'_0000.nii.gz'))
                        shutil.copyfile(os.path.join(root2, '3D',  'all_masks', study, patient, volume), os.path.join(root, 'Input voor nnUNet', 'labelsTr',  file_name+'.nii.gz'))
    
                                
            for patient in ListTest:
                if patient in os.listdir(os.path.join(root3, study)):
                    path2MRI = os.path.join(root3, study, patient)
                    MRI_volumes = os.listdir(path2MRI)
                    for volume in MRI_volumes:
                        file_name = patient + '_' + volume[:-7]
                        ListSlicesTest.append(file_name)    
                        shutil.copyfile(os.path.join(path2MRI, volume), os.path.join(root, 'Input voor nnUNet', 'imagesTs', file_name+'_0000.nii.gz'))
                        shutil.copyfile(os.path.join(path2MRI, volume), os.path.join(root, 'Input voor nnUNet', 'imagesTr', file_name+'_0000.nii.gz'))
                        shutil.copyfile(os.path.join(root2, '3D',  'all_masks', study, patient, volume), os.path.join(root, 'Input voor nnUNet', 'labelsTr', file_name+'.nii.gz'))
                        
    
                
    if dim == '2D': 
        ListEmptyMasks = []
        root3 = os.path.join(root2, '2D', 'all_masks')
        for study in os.listdir(root3):
            for patient in ListTrain:
                if patient in os.listdir(os.path.join(root3, study)):
                    path2Mask = os.path.join(root3, study, patient)
                    slices = os.listdir(path2Mask)
                    for slice in slices:
                        file_name = slice[:-7]
                        raw_mask = nib.load(os.path.join(path2Mask, slice))
                        mask = raw_mask.get_fdata()
                        if np.max(mask)>0:
                            ListSlicesTrain.append(file_name)
                            shutil.copyfile(os.path.join(root2, '2D', 'all_images', study, patient, slice), os.path.join(root, 'Input voor nnUNet', 'imagesTr', file_name+'_0000.nii.gz'))
                            shutil.copyfile(os.path.join(path2Mask, slice), os.path.join(root, 'Input voor nnUNet', 'labelsTr', slice))
                        else:
                            ListEmptyMasks.append(slice)
                            
        
        random.shuffle(ListEmptyMasks)
        ListEmptyMasks = ListEmptyMasks[:int(len(ListEmptyMasks)/4)]
        for study in os.listdir(root3):
            for patient in os.listdir(os.path.join(root3, study)):
                for slice in os.listdir(os.path.join(root3, study, patient)):
                    if slice in ListEmptyMasks:
                        file_name = slice[:-7]
                        ListSlicesTrain.append(file_name)
                        shutil.copyfile(os.path.join(root2, '2D', 'all_images', study, patient, slice), os.path.join(root, 'Input voor nnUNet', 'imagesTr', file_name+'_0000.nii.gz'))
                        shutil.copyfile(os.path.join(root3, study, patient, slice), os.path.join(root, 'Input voor nnUNet', 'labelsTr', slice))
                        
    
        for study in os.listdir(root3):
            for patient in ListTest:
                if patient in os.listdir(os.path.join(root3, study)):
                    path2Mask = os.path.join(root3, study, patient)
                    slices = os.listdir(path2Mask)
                    for slice in slices:
                        file_name = slice[:-7]
                        ListSlicesTest.append(file_name)
                        shutil.copyfile(os.path.join(root2, '2D', 'all_images', study, patient, slice), os.path.join(root, 'Input voor nnUNet', 'imagesTr', file_name+'_0000.nii.gz'))
                        shutil.copyfile(os.path.join(root2, '2D',  'all_images', study, patient, slice), os.path.join(root, 'Input voor nnUNet', 'imagesTs', file_name+'_0000.nii.gz'))
                        shutil.copyfile(os.path.join(root3, study, patient, slice), os.path.join(root, 'Input voor nnUNet', 'labelsTr', slice))
    
                    
    
    train_keys = np.array(ListSlicesTrain)
    val_keys = np.array(ListSlicesTest)
    
    splits[-1]['train'] = np.array(train_keys)
    splits[-1]['val'] = np.array(val_keys)
    
    finalpath = os.path.join(root, 'splits_final.pkl')
    
    #save split_final.pkl
    with open(finalpath, 'wb') as handle:
        pickle.dump(splits, handle, protocol=pickle.HIGHEST_PROTOCOL)