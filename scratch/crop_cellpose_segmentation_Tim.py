import os
import tifffile as tiff
import numpy as np
from tqdm import tqdm
from scipy.ndimage import binary_dilation, gaussian_filter, label
from skimage.measure import regionprops_table
import pandas as pd
from cellpose import models
from scipy import ndimage as ndi
from concurrent.futures import ThreadPoolExecutor, as_completed
import pstats
import cProfile
import logging
from datetime import datetime



###########################
######### Parameters ######
###########################
image_info = '/workspaces/ChronoType/DINO_Pretraining/data/raw/metadata/Chronotype_HCP_WP3_imageInfo_with_subclass.csv'
output_features_file = "/workspaces/ChronoType/DINO_Pretraining/data/raw/metadata/CellPose_features.csv"
# Generate a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"/workspaces/ChronoType/DINO_Pretraining/data/processed/logscrop_cellpose_segmentation_{timestamp}.log"

batch_size = 100

######################################
######### Load model and data ######
######################################

# Load image info
image_info = pd.read_csv(image_info)

# add columns to image_info
image_info['Cell_Area'] = np.nan
image_info['Cell_Eccentricity'] = np.nan
image_info['Cell_MajorAxisLength'] = np.nan
image_info['CellMeanIntensity_APC'] = np.nan
image_info['CellMeanIntensity_Brightfield'] = np.nan
image_info['CellMeanIntensity_DAPI'] = np.nan
image_info['CellMeanIntensity_GFP'] = np.nan
image_info['CellMeanIntensity_PE'] = np.nan

# Initialize Cellpose model for nucleus and cell segmentation
nucleus_model = models.CellposeModel(gpu=True, model_type='nuclei')

######################################
######### Classes and Functions ######
######################################


def simple_watershed_scipy(image, threshold, size):
    # Step 1: Apply Gaussian smoothing (optional but helps reduce noise)
    smoothed_image = ndi.gaussian_filter(image, sigma=1)
    
    # Step 2: Apply a threshold using the 25% quantile
    threshold_value = np.percentile(smoothed_image, threshold)
    binary_mask = smoothed_image > threshold_value
    
    # Step 3: Compute the distance transform
    distance_transform = ndi.distance_transform_edt(binary_mask)
    
    # Step 4: Find local maxima to use as markers for the watershed algorithm
    local_maxima = ndi.label(ndi.maximum_filter(distance_transform, size=size) == distance_transform)[0]
    
    # Step 5: Apply the watershed algorithm
    watershed_result = ndi.watershed_ift(binary_mask.astype(np.uint8), local_maxima)
    
    return watershed_result

    # Feature extraction function
def extract_features(masks, imgs, image_names):

    # Initialize dictionary to hold the features for a single row
    batch_feature_pd = pd.DataFrame()
    channel_names = ['APC', 'Brightfield', 'DAPI', 'GFP', 'PE']

    for i, mask in enumerate(masks):
        feature_dict = {'ImageName': image_names[i]}
        image = imgs[i]
        apc, brightfield, dapi, gfp, pe = np.split(image, 5, axis=-1)
            
        channels = [np.squeeze(apc), np.squeeze(brightfield), np.squeeze(dapi), np.squeeze(gfp), np.squeeze(pe)]

        labeled_mask, num_features = label(mask)
        if num_features == 0:
            return pd.DataFrame()  # Return empty DataFrame if no features found

    
    
        # Get basic properties (e.g., area of the segmented region)
        properties = regionprops_table(labeled_mask, properties=['area', 'eccentricity', 'major_axis_length'])
    
        # Summarize properties across all labels (e.g., sum of areas)
        feature_dict['Cell_Area'] = np.sum(properties['area'])
        feature_dict['Cell_Eccentricity'] = np.sum(properties['eccentricity'])
        feature_dict['Cell_MajorAxisLength'] = np.sum(properties['major_axis_length'])

        # Extract mean intensity for each channel and store as separate columns
        for i, channel in enumerate(channels):
            intensity_props = regionprops_table(labeled_mask, intensity_image=channel, properties=['mean_intensity'])
        
            # Calculate the average mean intensity for all cells in this channel
            feature_dict[f'CellMeanIntensity_{channel_names[i]}'] = np.mean(intensity_props['mean_intensity'])
    
        # Append the feature dictionary to the list
        feature_df = pd.DataFrame([feature_dict])
        batch_feature_pd = pd.concat([batch_feature_pd, feature_df], ignore_index=True)
    return batch_feature_pd

def get_images(file_names):
    imgs = []
    image_5_channels = []
    for files in file_names:
        try:
            # Load the image
            image = tiff.imread(files)
            image_5_channels.append(image)    
            # Split the image into channels
            apc, brightfield, dapi, gfp, pe = np.split(image, 5, axis=-1)
            image = apc + dapi + gfp + pe
            
            imgs.append(image)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    return imgs,image_5_channels

def refine_mask(nucleus_masks,image_5_channels, threshold=0.5):
    final_masks = []
    for i, nucleus_mask in enumerate(nucleus_masks):
        center_pixel = nucleus_mask[25, 25]

        selected_cell_mask = (nucleus_mask == center_pixel).astype(np.uint8)
    
        # Dilate mask by XX pixels to include more of the cell
        dilated_mask = binary_dilation(selected_cell_mask, iterations=5).astype(np.uint8)
    
        # If there is no mask just use a regular watershed as defined above - works quite well
        if center_pixel == 0:
            image = image_5_channels[i]
            apc, brightfield, dapi, gfp, pe = np.split(image, 5, axis=-1)
            nucleus_mask = simple_watershed_scipy(apc+dapi+gfp+pe, threshold=50, size=3)
            nucleus_mask = np.squeeze(nucleus_mask)
            center_pixel = nucleus_mask[25, 25]
            selected_cell_mask = (nucleus_mask == center_pixel).astype(np.uint8)
            dilated_mask = binary_dilation(selected_cell_mask, iterations=1).astype(np.uint8) # since the watershed is a bit more permissive I don't do the dilation as strong, but anyway irrelevant for the feature derivation

        # Apply Gaussian filter to smoothen the dilated mask
        smoothed_mask = gaussian_filter(dilated_mask.astype(float), sigma=5)
    
        # Re-binarize the smoothed mask
        final_mask = (smoothed_mask > 0.5).astype(np.uint16)

        for threshold, size, iterations in [(50, 5, 1), (60, 5, 3)]:
            if final_mask[25, 25] == 0:
                print(f'Still no mask - changing threshold to {threshold} and size to {size}')
                nucleus_mask = simple_watershed_scipy(apc + dapi + gfp + pe, threshold, size)
                nucleus_mask = np.squeeze(nucleus_mask)
                center_pixel = nucleus_mask[25, 25]
                selected_cell_mask = (nucleus_mask == center_pixel).astype(np.uint8)
                dilated_mask = binary_dilation(selected_cell_mask, iterations=iterations).astype(np.uint8)
                smoothed_mask = gaussian_filter(dilated_mask.astype(float), sigma=5)
                final_mask = (smoothed_mask > 0.5).astype(np.uint16)

        if final_mask[25, 25] == 0:
            print('No Mask found - keeping the whole crop!')
            final_mask = np.ones(dapi.shape[:2], dtype=np.uint8)

        final_masks.append(final_mask)
    return final_masks



def process_image(filenames):
    
    imgs, image_5_channels = get_images(filenames)

    # Perform nucleus segmentation on each crop - I've found that the addition of all channels gives the best cell segmentation
    nucleus_masks = nucleus_model.eval(imgs, flow_threshold=0, diameter=None, resample=1, channels=[0,0])[0]

    masks = refine_mask(nucleus_masks,image_5_channels)
    
    # Extract features from each channel using the final mask - here you can decide which mask to choose
    features_df = extract_features(masks, image_5_channels, filenames)
    # print shape of image_5_channels
    
    return features_df


    

#####################
######### main ######
#####################

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file)])
    logger = logging.getLogger(__name__)

    all_features = pd.DataFrame()
    image_paths = image_info['Image Path']
    length_pd = len(image_paths)

    logger.info("Starting image processing")

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_image, image_paths[i:i + batch_size].tolist()) for i in range(0, length_pd, batch_size)]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches", leave=False):
            try:
                features_df = future.result()
                if not features_df.empty:
                    all_features = pd.concat([all_features, features_df], ignore_index=True)
                logger.info(f"Processed {len(all_features)} images out of {length_pd}")
            except Exception as e:
                logger.error(f"Error processing batch: {e}")

    # Save the features to a CSV file
    all_features.to_csv(output_features_file, index=False)
    logger.info(f"Features saved to {output_features_file}")

