# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 19:09:21 2021

@author: s132525
"""

import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import code

# Create and save MRI image files and XML files as 2D and 3D nifti files. 

def XML_to_3DNifti(root, check_masks):
    root1 = os.path.join(root, 'raw_data')
    root2 = os.path.join(root, 'model_data')
    if not os.path.exists(root2):
        os.mkdir(root2)
    if not os.path.exists(os.path.join(root2,'3D')):
        os.mkdir(os.path.join(root2,'3D'))
    if not os.path.exists(os.path.join(root2,'3D', 'all_masks')):
        os.mkdir(os.path.join(root2,'3D', 'all_masks'))
    FDM = os.path.join(root2,'3D', 'all_masks')
    if not os.path.exists(os.path.join(root2, '3D', 'all_images')):
        os.mkdir(os.path.join(root2, '3D', 'all_images'))        
    FDMri = os.path.join(root2,'3D', 'all_images')     
    studies = os.listdir(root1)
    for study in studies:
        rootM = os.path.join(root1, study, 'Mask')
        rootMRI = os.path.join(root1, study, 'MRI')
        MaskP = os.listdir(rootM)
        FDMask = os.path.join(FDM, study)
        FDMRI = os.path.join(FDMri, study)
        if not os.path.exists(FDMRI):
            os.mkdir(FDMRI)
        if not os.path.exists(FDMask):
            os.mkdir(FDMask)
        for patient in MaskP:
            rootMask = os.path.join(rootM, patient)
            tplist = os.listdir(rootMask)
            if not os.path.exists(os.path.join(FDMask, patient)):
                os.mkdir(os.path.join(FDMask, patient))
            if not os.path.exists(os.path.join(FDMRI, patient)):
                os.mkdir(os.path.join(FDMRI, patient))
            for tp in tplist:
                im_path = os.path.join(rootMRI, patient)
                if os.listdir(im_path)[1].find('.img')>0:                                       #If the MRI file is an .img file, it will run these lines
                    im_name = os.path.join(im_path, tp + str('.img'))
                    maskorientation = 'mirrored'
                    mriorientation = 'normal'
                
                else:           
                                                                                                #If the MRI file is a nifti file, it will run these lines
                    if os.listdir(im_path)[1].find('.gz')>0:
                        im_name = os.path.join(im_path, tp + str('.nii.gz'))
                        maskorientation = 'normal' 
                        mriorientation = 'normal'
                    else: 
                        im_name = os.path.join(im_path, tp + str('.nii'))
                        maskorientation = 'mirrored'
                        mriorientation = 'mirrored'
                    
                autoData = nib.load(im_name)
                if len(autoData.shape) == 3:
                    dataNii = autoData.get_fdata()
                else:
                    dataNii = autoData.get_fdata().squeeze()
                
                if mriorientation == 'mirrored':
                    dataNii = np.flip(dataNii,1)
                Number_of_slices = autoData.shape[2]
                aff = autoData.affine
                VoxelDim = [aff[0,0], aff[1,1], aff[2,2]]
                xmls = os.path.join(rootMask, tp)
                xml_list = os.listdir(xmls)
                Coords = {}
                for xml in xml_list:
                    xml_path = os.path.join(xmls, xml)
                    if xml_path[-3:]=='xml':
                        inf = open(xml_path)
                        all_info = inf.read()
                        inf.close()
                        
                        if all_info.find('<Slice-number>')!=-1:                                    #The slice number is indicated in the XML file, coordinates here are 320x220
                            slices_list = all_info.split('<Slice-number>')
                            slices = []
                            extracount = 0
                            for slice_info in slices_list:
                                if slice_info[slice_info.find('</'):slice_info.find('>')] == '</Slice-number':
                                    slice_number = int(slice_info[:slice_info.find('</')])
                                    coord_info = slice_info[slice_info.find('</Slice-number>'): slice_info.find('<\t/Contour>')].split('\n')
                                    slices.append(slice_number)
                                    
                                    if slices.count(slice_number)==1:
                                        slice_number = str(slice_number).zfill(2)
                                        Coords[slice_number] = []
                                    else:
                                        extracount = extracount+1
                                        slice_number = str(slice_number).zfill(2)+'_'+str(extracount)
                                        Coords[slice_number] = []
                                        
                                        
                                    for item in coord_info:
                                        items = item.strip().split('>')
                                        if items[0] =='<Pt':
                                            these_coords =  items[1].split('<')[0].split(',')
                                            if maskorientation == 'normal':
                                                Coords[slice_number].append( (float(these_coords[1]), float(these_coords[0]), ))    
                                            else:
                                                Coords[slice_number].append( (224-float(these_coords[1]), float(these_coords[0]), ))    
                                    Coords[slice_number].append(Coords[slice_number][0])                     
                        else:                                                                      #The slice number and coordinates are 3D
                            slices_list = all_info.split('<Contour>')
                            slices_list.pop(0)
                            slices = []
                            extracount = 0
                            for slice_info in slices_list:
                                coord_info = slice_info.split('</Pt')
                                coord_info.pop(-1)
                                if study == 'Midi':                                   
                                    slice_number = round((Number_of_slices)*2-float(coord_info[0].strip().split('<Pt>')[1].split(',')[1])/VoxelDim[2]-2) #In this study the 3D coordinates in the slices are shifted by 2: Get slicenumbers between 1-24
                                else:
                                    slice_number = round((Number_of_slices)*2-float(coord_info[0].strip().split('<Pt>')[1].split(',')[1])/VoxelDim[2])
                                slices.append(slice_number)
                                
                                if slices.count(slice_number)==1:
                                    slice_number = str(slice_number).zfill(2)
                                    Coords[slice_number] = []
                                else:
                                    extracount = extracount+1
                                    slice_number = str(slice_number).zfill(2)+'_'+str(extracount)
                                    Coords[slice_number] = []
                                        
                                
                                for item in coord_info:
                                    items = item.strip().split('<Pt>')[1].split(',')
                                    if maskorientation == 'normal':
                                        Coords[slice_number].append( (float(items[2])/VoxelDim[0], float(items[0])/VoxelDim[1]) )
                                    else:
                           
                                         Coords[slice_number].append( (224-float(items[2])/VoxelDim[0], float(items[0])/VoxelDim[1]) )
                                Coords[slice_number].append(Coords[slice_number][0])   
                
                lastslice = ''    
                mask_3D = np.zeros((320, 224, Number_of_slices), dtype=np.uint8)     
                for slice_number in dict.keys(Coords):
                    # Create 2D mask
                    if len(Coords[slice_number])>1:
                        if slice_number[0:2]==lastslice:
                            mask_2D = Image.new('L', (224,320), 0)
                            ImageDraw.Draw(mask_2D).polygon(Coords[slice_number], outline=1, fill=1)
                            mask_3D[:,:,int(slice_number[0:2])] = mask_3D[:,:, int(slice_number[0:2])]+np.array(mask_2D)
                        else:
                            mask_2D = Image.new('L', (224,320,), 0)
                            ImageDraw.Draw(mask_2D).polygon(Coords[slice_number], outline=1, fill=1)
                            mask_3D[:,:,int(slice_number[0:2])] = np.array(mask_2D)
                        lastslice = slice_number[0:2]
                        
                    else:
                        print(patient, tp, slice_number)
                
                for i in range(0,dataNii.shape[2]):            
                    slice_1 = dataNii[:, :, i]
                    slice_2 = mask_3D[:, :, i]
                    plt.figure()
                    plt.imshow(slice_1, cmap='gray')
                    plt.imshow(slice_2, alpha=0.5)
                    plt.title(str(patient + ' ' + tp + ' ' + str(i)))
                    plt.show()
                
                MriTPnii = nib.Nifti1Image(dataNii, affine=aff, header = autoData.header)
                MaskTPnii = nib.Nifti1Image(mask_3D, affine = aff)
                
                nib.save(MaskTPnii, os.path.join(FDMask, patient, str(tp+'.nii.gz'))) 
                nib.save(MriTPnii, os.path.join(FDMRI, patient, str(tp+'.nii.gz'))) 
                if check_masks == True:
                    code.interact(banner='Paused. Press ^D (Ctrl+D) to continue.', local=globals())
    
def Nifti2D(root):
    root1 = os.path.join(root, 'raw_data')
    root2 = os.path.join(root, 'model_data')
    FDM = os.path.join(root2, '3D', 'all_masks')        
    FDMri = os.path.join(root2,'3D', 'all_images')
    studies = os.listdir(root1)
    
    if not os.path.exists(os.path.join(root2, '2D')):
        os.mkdir(os.path.join(root2, '2D'))
    if not os.path.exists(os.path.join(root2, '2D', 'all_images')):
        os.mkdir(os.path.join(root2, '2D', 'all_images'))
    if not os.path.exists(os.path.join(root2, '2D', 'all_masks')):
        os.mkdir(os.path.join(root2, '2D', 'all_masks'))
    FDMri2D = os.path.join(root2, '2D', 'all_images')
    FDM2D = os.path.join(root2, '2D', 'all_masks')
    
    
    for study in studies:
        if not os.path.exists(os.path.join(FDMri2D, study)):
            os.mkdir(os.path.join(FDMri2D, study))
        if not os.path.exists(os.path.join(FDM2D, study)):
            os.mkdir(os.path.join(FDM2D, study))
        patients = os.listdir(os.path.join(FDMri, study))
        for patient in patients:
            if not os.path.exists(os.path.join(FDMri2D, study, patient)):
                os.mkdir(os.path.join(FDMri2D, study, patient))
            if not os.path.exists(os.path.join(FDM2D, study, patient)):
                os.mkdir(os.path.join(FDM2D, study, patient))
            tps = os.listdir(os.path.join(FDMri, study, patient))
            for tp in tps:
                path1 = os.path.join(FDMri, study, patient, tp)
                path2 = os.path.join(FDM, study, patient, tp)
                d1 = nib.load(path1)
                mri = d1.get_fdata()
                d2 = nib.load(path2)
                mask = d2.get_fdata()
                aff = d1.affine
                tpname = tp[0:-7]
                for i in range(0,mri.shape[2]):        
                    slice_1 = mri[:, :, i]
                    reshaped1 = slice_1.reshape(320,224,1)
                    slice_2 = mask[:, :, i]
                    reshaped2 = slice_2.reshape(320,224,1)
                    MRI = nib.Nifti1Image(reshaped1, affine=aff)
                    MASK = nib.Nifti1Image(reshaped2, affine = aff)
                    nib.save(MRI, os.path.join(FDMri2D, study, patient, str(patient+'_'+tpname+'_'+str(i)+'.nii.gz'))) 
                    nib.save(MASK, os.path.join(FDM2D, study, patient, str(patient+'_'+tpname+'_'+str(i)+'.nii.gz'))) 
                    
            
    
