import os
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import transform as skTrans
import pandas as pd

#------------------------------------------------- Data-------------------------------------------------#
# Test
test_data = "/data/Datasets/stroke/research_subsets/MICCAI24/test"

# Valid
valid_data = "/data/Datasets/stroke/research_subsets/MICCAI24/valid"

# Train
train_data = "/data/Datasets/stroke/research_subsets/MICCAI24/train"

#------------------------------ Big Lession and volume function ---------------------------------------#

def biggest_lession(adc_mask_np):
    biggest_lesion_slice_idx = None
    biggest_lesion_pixel_count = 0
    slices_with_lession = []
    for idx, mask_slice in enumerate(adc_mask_np.transpose(2, 0, 1)):
        # Slice with lesion.
        slice_pixel_count = np.count_nonzero(mask_slice)
        if slice_pixel_count > 0:
            slices_with_lession.append(idx)
            if slice_pixel_count > biggest_lesion_pixel_count:
                biggest_lesion_pixel_count = slice_pixel_count
                biggest_lesion_slice_idx = idx
    return biggest_lesion_slice_idx, biggest_lesion_pixel_count, slices_with_lession

##Calcular el volumen
def get_voxel_volume_in_ml(nii):
    """Converts mm3 to ml."""
    mm3 = np.prod(nii.header.get_zooms())
    return mm3 / 1000.0

#----------------------------------------------Create dataset------------------------------------------#

def create_dataset(dataset):
    non_ADC_segmentation = []; ADC_segmentation = []; radiologist_annotation = []; 
    patched_mask = []; ADC_patched_image = []; ADC_masked_image = []; complete_mask = [];
    list_biggest_lesion_slice_idx = [];
    ADC_resized_list = []
    
    # Clinical information
    treatment_list = []; age_list = []; nihss_list = []; time_list = []; territory_list = []; 
    volume_ml_ADC = [];

    for i in os.listdir(dataset): 
        patient= os.path.join(dataset, i) 

        # Without lesion
        value=1 
        lesion_information= os.path.join(dataset, i,f"{i}_lesion_information.json")
        with open(lesion_information) as d:
               data_l = json.load(d)
        if(len(data_l['lesion_slices'])==0): value = 0; non_ADC_segmentation.append(patient)

        elif(value ==1):   
            # Clinical information
            clinical_information= os.path.join(dataset, i,f"{i}_clinical_variables.json") 
            print(patient)
            print(i)
            with open(clinical_information) as d:
               data = json.load(d)
                
            if ((data['Tratamiento realizado'] == '2, 3') or (data['Tratamiento realizado'] == '2,3')):
                treatment = 2.0
            else:
                treatment = float(data['Tratamiento realizado'])
                
            if data['Territorio arterial comprometido'] == '10,11':
                territory = 10.0
            elif data['Territorio arterial comprometido'] == '3,4':
                territory = 3.0
            elif data['Territorio arterial comprometido'] == '2,1':
                territory = 1.0
            elif ((data['Territorio arterial comprometido'] == '2,3') or (data['Territorio arterial comprometido'] == '2, 3') or (data['Territorio arterial comprometido'] == '2,14')):
                territory = 2.0
            elif data['Territorio arterial comprometido'] == '4,5':
                territory = 4.0
            elif data['Territorio arterial comprometido'] == '7,8':
                territory = 7.0
            elif data['Territorio arterial comprometido'] == '12,14':
                territory = 12.0
            elif data['Territorio arterial comprometido'] == '13,12':
                territory = 12.0
            else:
                territory = float(data['Territorio arterial comprometido'])
            treatment = treatment-1.0
            if(treatment == 3.0): value = 0
        
            if(value==1):
                treatment_list.append(treatment)
                territory_list.append(territory-1.0)
                age_list.append(data['Edad'])
                nihss_list.append(data['NIHSS ingreso'])
                time_list.append(data['(Minutos)Tiempo desde inicio de sintomas hasta ingreso a urgencias'])
                print('Tratamiento:', territory_list)
                masks_dir = os.path.join(patient, "Masks", "AnaAraujo")
                if(os.path.isdir(masks_dir)): mask_list = os.listdir(masks_dir); 
                else: mask_list = []
                print(mask_list)
                
        
                # Take Mask (First Ana Teresa) 
                
                if(len(mask_list) > 0): radiologist_annotation.append("AnaAraujo")
                if((len(mask_list) == 0)): 
                    masks_dir = os.path.join(patient, "Masks", "Andres")
                    mask_list = os.listdir(masks_dir);
                    if(len(mask_list) > 0): radiologist_annotation.append("Andres")
                if((len(mask_list) == 0)): masks_dir = os.path.join(patient, "Masks", "Daniel"); mask_list = os.listdir(masks_dir); radiologist_annotation.append("Daniel");  
                
        
        
                ADC_ind = [ind for ind, list_ in enumerate(mask_list) if ("ADC" in list_)]  # Search ADC
                if(len(ADC_ind) == 0): non_ADC_segmentation.append(patient)
    
                
                if(1==1):
                    ADC_segmentation.append(patient)
    
                    # Getting the volume and also the one with the highest lession
                    adc_mask_path = os.path.join(masks_dir, 'ADC.nii.gz')
                    adc_mask_nii = nib.load(adc_mask_path)
                    adc_mask_np = adc_mask_nii.get_fdata()
    
                    # Getting the biggest lession on the ADC volume
                    adc_pixel_count = np.count_nonzero(adc_mask_np)
                    print("adc_pixel_count: ", adc_pixel_count)
                    biggest_lesion_slice_idx, biggest_lesion_pixel_count, slices_with_lession = biggest_lession(adc_mask_np)
                    list_biggest_lesion_slice_idx.append(biggest_lesion_slice_idx)
    
                    # Getting lesion volume
                    volume_ml_ADC.append(get_voxel_volume_in_ml(adc_mask_nii)*adc_pixel_count)
    
                    # Find the contours from the ADC mask
    
                    slice_mask = adc_mask_np[:,:,biggest_lesion_slice_idx].astype(np.uint8)
                    print("biggest_lesion_slice_idx", biggest_lesion_slice_idx, slice_mask.shape)
                    plt.imshow(slice_mask, cmap="gray") 
                    plt.show(); plt.close()
                    complete_mask.append(slice_mask)
                    
                        
                    contours, hierarchy = cv.findContours(slice_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
                    big_contour = max(contours, key=cv.contourArea)
                    contr = slice_mask.copy()
                    contr = cv.drawContours(contr, [big_contour], 0, (0,0,255), 2)
                    x,y,w,h = cv.boundingRect(contours[0])
                    contour_lession = slice_mask[y-10:y+h+10,x-10:x+w+10]
                    print("How big is the lesion?: ", contour_lession.shape)
    
                    resized_contour_lession =  skTrans.resize(contour_lession, (32,32,1), order=1, preserve_range=True)
                    patched_mask.append(resized_contour_lession)
    
                    # Imagenology MRI
                    patient_studies = os.listdir(patient)
                    print("patient_studies: ", patient_studies)
                    ADC_ind = [ind for ind, list_ in enumerate(patient_studies) if ("ADC" in list_)]
                    print("ADC_ind:         ", ADC_ind)
                    
                    # Getting the ADC volume
                    ADC_path = os.path.join(patient, patient_studies[ADC_ind[0]])
                    ADC_image_path = os.path.join(ADC_path, "ADC_brain_extracted.nii.gz")
                    ADC_image_nii = nib.load(ADC_image_path)
                    ADC_image_np = ADC_image_nii.get_fdata()
                    ADC_image_np_norm = (ADC_image_np - np.mean(ADC_image_np)) / np.std(ADC_image_np)
                    print("ADC_image_np: ", ADC_image_np.shape)
    
                    
                    slice_ADC_image = ADC_image_np_norm[:,:,biggest_lesion_slice_idx].astype(np.uint8)
                    plt.imshow(slice_ADC_image, cmap="gray") 
                    plt.show(); plt.close()
    
                    ADC_lession = slice_ADC_image[y-10:y+h+10,x-10:x+w+10]
    
                    
                    ADC_resized =  skTrans.resize(ADC_lession, (32,32,1), order=1, preserve_range=True)
                    
                    print("ADC_resized", ADC_resized.shape)
                    ADC_resized_list.append(ADC_resized)



                plt.imshow(ADC_resized, cmap="gray") 
                plt.show(); plt.close()
                # plt.imshow(DWI_resized, cmap="gray") 
                # plt.show(); plt.close()
                APIS_tabular = pd.DataFrame(list(zip(age_list,nihss_list, time_list,treatment_list,volume_ml_ADC )), 
                                      columns = ['age', 'NIHSS', 'time','treatment', 'volume'])
                APIS_tabular['time'] = pd.to_numeric(APIS_tabular['time'], errors='coerce')
                APIS_tabular['time'] = APIS_tabular['time'] / 60
                # APIS_tabular = APIS_tabular.drop(97).reset_index(); del APIS_tabular['index']
                territorio_d = pd.DataFrame(territory_list,  columns = ['Territorio'])
    return APIS_tabular, territorio_d, ADC_resized_list

train_dataset, territorio_train, ADC_resized_list_train = create_dataset(train_data)
valid_dataset, territorio_valid, ADC_resized_list_valid = create_dataset(valid_data)
test_dataset, territorio_test, ADC_resized_list_test = create_dataset(test_data)

# Train
X_train_img = np.asarray(ADC_resized_list_train)
y_train_lab = train_dataset.values[:, -2].flatten()
X_train_txt = train_dataset.values[:, :-2]  
X_train_terr = territorio_train.values[:, :] 

# Valid 
X_valid_img = np.asarray(ADC_resized_list_valid)
y_valid_lab = valid_dataset.values[:, -2].flatten()
X_valid_txt = valid_dataset.values[:, :-2]  
X_valid_terr = territorio_valid.values[:, :] 

# Test
X_test_img = np.asarray(ADC_resized_list_test)
y_test_lab = test_dataset.values[:, -2].flatten()
X_test_txt = test_dataset.values[:, :-2]  
X_test_terr = territorio_test.values[:, :] 

# Concatenate
X_values_img = np.concatenate((X_train_img, X_valid_img, X_test_img), axis=0)
y_values_lab = np.concatenate((y_train_lab, y_valid_lab, y_test_lab), axis=0).reshape(-1, 1).astype(float)
X_values_txt = np.concatenate((X_train_txt, X_valid_txt, X_test_txt), axis=0)
X_values_terr = np.concatenate((X_train_terr, X_valid_terr, X_test_terr), axis=0)