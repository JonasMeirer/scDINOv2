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
from scipy.ndimage import binary_erosion
from skimage.feature import graycomatrix
from skimage.filters import gabor


###########################
######### Parameters ######
###########################

input_folder = "D:/iPSC/YS_Phenoplex/Trial/"
output_features_file =  "D:/iPSC/YS_Phenoplex/Trial/CellPose_features.csv"
# Generate a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"D:/iPSC/YS_Phenoplex/Trial/logscrop_cellpose_segmentation_{timestamp}.log"

batch_size = 100

######################################
######### Load model and data ######
######################################


# Initialize Cellpose model for nucleus and cell segmentation
nucleus_model = models.CellposeModel(gpu=True, model_type='nuclei')

######################################
######### Classes and Functions ######
######################################

def build_image_info(input_folder):
    rows = []

    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".tiff"):
                input_path = os.path.join(root, filename)
                rel_path = os.path.relpath(input_path, input_folder)

                rows.append({
                    "ImageName": filename,
                    "Image Path": input_path,
                    "Relative Path": rel_path
                })

    image_info = pd.DataFrame(rows)
    return image_info

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


def keep_largest_object(mask):
    mask = np.squeeze(mask).astype(bool)
    labeled_mask, num_features = label(mask)
    if num_features == 0:
        return None

    if num_features == 1:
        return mask.astype(np.uint8)

    sizes = ndi.sum(mask, labeled_mask, index=np.arange(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    return (labeled_mask == largest_label).astype(np.uint8)


def get_edge_mask(mask):
    mask = mask.astype(bool)
    eroded = binary_erosion(mask)
    edge = mask & (~eroded)
    if edge.sum() == 0:
        edge = mask
    return edge.astype(np.uint8)


def safe_mad(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    med = np.median(values)
    return np.median(np.abs(values - med))


def safe_stats(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            "IntegratedIntensity": np.nan,
            "LowerQuartileIntensity": np.nan,
            "MADIntensity": np.nan,
            "MaxIntensity": np.nan,
            "MeanIntensity": np.nan,
            "MedianIntensity": np.nan,
            "MinIntensity": np.nan,
            "StdIntensity": np.nan,
            "UpperQuartileIntensity": np.nan,
        }

    return {
        "IntegratedIntensity": np.sum(values),
        "LowerQuartileIntensity": np.percentile(values, 25),
        "MADIntensity": safe_mad(values),
        "MaxIntensity": np.max(values),
        "MeanIntensity": np.mean(values),
        "MedianIntensity": np.median(values),
        "MinIntensity": np.min(values),
        "StdIntensity": np.std(values),
        "UpperQuartileIntensity": np.percentile(values, 75),
    }


def mass_displacement(mask, intensity_image):
    mask = mask.astype(bool)
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return np.nan

    geom_centroid = coords.mean(axis=0)

    intensities = intensity_image[mask].astype(float)
    total_intensity = intensities.sum()
    if total_intensity <= 0:
        return 0.0

    weighted_centroid = np.average(coords, axis=0, weights=intensities)
    return np.linalg.norm(weighted_centroid - geom_centroid)


def add_intensity_features(feature_dict, mask, channel, compartment, channel_name):
    mask = mask.astype(bool)
    edge_mask = get_edge_mask(mask).astype(bool)

    vals = channel[mask]
    edge_vals = channel[edge_mask]

    stats_main = safe_stats(vals)
    stats_edge = safe_stats(edge_vals)

    for stat_name, value in stats_main.items():
        feature_dict[f"{compartment}_Intensity_{stat_name}_{channel_name}"] = value

    for stat_name, value in stats_edge.items():
        feature_dict[f"{compartment}_Intensity_{stat_name}Edge_{channel_name}"] = value

    feature_dict[f"{compartment}_Intensity_MassDisplacement_{channel_name}"] = mass_displacement(mask, channel)


def quantize_image_for_texture(channel, mask, levels=32):
    """
    Quantise intensities inside the mask to [0, levels-1].
    Outside-mask pixels are set to 0.
    """
    mask = mask.astype(bool)
    out = np.zeros_like(channel, dtype=np.uint8)

    vals = channel[mask].astype(float)
    if vals.size == 0:
        return out

    vmin = vals.min()
    vmax = vals.max()

    if vmax <= vmin:
        out[mask] = 0
        return out

    scaled = (channel.astype(float) - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0, 1)
    quantized = np.floor(scaled * (levels - 1)).astype(np.uint8)
    out[mask] = quantized[mask]
    return out


def _entropy(p):
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return -np.sum(p * np.log2(p))


def haralick_features_from_glcm(P):
    """
    P should be a normalised 2D GLCM for one distance/angle.
    """
    eps = 1e-12
    P = P.astype(float)
    P = P / (P.sum() + eps)

    n = P.shape[0]
    i, j = np.indices((n, n))

    px = P.sum(axis=1)
    py = P.sum(axis=0)

    mux = np.sum(i[:, 0] * px)
    muy = np.sum(j[0, :] * py)
    sigx = np.sqrt(np.sum(((i[:, 0] - mux) ** 2) * px))
    sigy = np.sqrt(np.sum(((j[0, :] - muy) ** 2) * py))

    asm = np.sum(P ** 2)
    contrast = np.sum((i - j) ** 2 * P)

    if sigx > 0 and sigy > 0:
        correlation = np.sum((i - mux) * (j - muy) * P) / (sigx * sigy)
    else:
        correlation = 0.0

    idm = np.sum(P / (1.0 + (i - j) ** 2))
    entropy = _entropy(P)

    mu = np.sum(i * P)
    variance = np.sum(((i - mu) ** 2) * P)

    # p_xplusy
    p_xplusy = np.zeros(2 * n - 1)
    for k in range(2 * n - 1):
        p_xplusy[k] = np.sum(P[(i + j) == k])

    # p_xminusy
    p_xminusy = np.zeros(n)
    for k in range(n):
        p_xminusy[k] = np.sum(P[np.abs(i - j) == k])

    sum_average = np.sum(np.arange(2 * n - 1) * p_xplusy)
    sum_entropy = _entropy(p_xplusy)
    sum_variance = np.sum(((np.arange(2 * n - 1) - sum_average) ** 2) * p_xplusy)

    difference_entropy = _entropy(p_xminusy)
    difference_variance = np.var(np.repeat(np.arange(n), np.round(p_xminusy * 1000).astype(int))) if np.sum(p_xminusy) > 0 else 0.0

    HX = _entropy(px)
    HY = _entropy(py)
    HXY = entropy

    px_py = np.outer(px, py)
    HXY1 = -np.sum(P[P > 0] * np.log2(px_py[P > 0] + eps))
    HXY2 = _entropy(px_py)

    info_meas1 = (HXY - HXY1) / max(HX, HY, eps)
    info_meas2 = np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * (HXY2 - HXY))))

    return {
        "AngularSecondMoment": asm,
        "Contrast": contrast,
        "Correlation": correlation,
        "DifferenceEntropy": difference_entropy,
        "DifferenceVariance": difference_variance,
        "Entropy": entropy,
        "InfoMeas1": info_meas1,
        "InfoMeas2": info_meas2,
        "InverseDifferenceMoment": idm,
        "SumAverage": sum_average,
        "SumEntropy": sum_entropy,
        "SumVariance": sum_variance,
        "Variance": variance,
    }


def add_texture_features(feature_dict, mask, channel, compartment, channel_name, distance=6, levels=32):
    """
    Approximates CellProfiler-style Haralick texture features.
    Uses 4 angles and averages them.
    """
    mask = mask.astype(bool)

    if mask.sum() < 10:
        texture_names = [
            "AngularSecondMoment", "Contrast", "Correlation",
            "DifferenceEntropy", "DifferenceVariance", "Entropy",
            "InfoMeas1", "InfoMeas2", "InverseDifferenceMoment",
            "SumAverage", "SumEntropy", "SumVariance", "Variance",
            "Gabor"
        ]
        for name in texture_names:
            feature_dict[f"{compartment}_Texture_{name}_{channel_name}_{distance}"] = np.nan
        return

    quantized = quantize_image_for_texture(channel, mask, levels=levels)

    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    crop = quantized[r0:r1, c0:c1]

    glcm = graycomatrix(
        crop,
        distances=[distance],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=levels,
        symmetric=True,
        normed=True
    )

    collected = {}
    for angle_idx in range(glcm.shape[3]):
        P = glcm[:, :, 0, angle_idx]
        feats = haralick_features_from_glcm(P)
        for k, v in feats.items():
            collected.setdefault(k, []).append(v)

    for k, vals in collected.items():
        feature_dict[f"{compartment}_Texture_{k}_{channel_name}_{distance}"] = np.mean(vals)

    # Approximate Gabor texture feature
    channel_float = channel.astype(np.float32)
    vals = channel_float[mask]
    if vals.size == 0 or vals.max() <= vals.min():
        gabor_mean = np.nan
    else:
        norm_channel = (channel_float - vals.min()) / (vals.max() - vals.min() + 1e-12)
        real, imag = gabor(norm_channel, frequency=0.1)
        gabor_mag = np.sqrt(real**2 + imag**2)
        gabor_mean = np.mean(gabor_mag[mask])

    feature_dict[f"{compartment}_Texture_Gabor_{channel_name}_{distance}"] = gabor_mean



    # Feature extraction function
def extract_features(masks, imgs, image_names):

    batch_feature_pd = pd.DataFrame()

    # Intensity features were requested only for these channels
    cell_intensity_channels = ['APC', 'Brightfield','DAPI', 'GREEN', 'PE']

    # Texture features were requested for all 5 channels
    texture_channels = ['APC', 'Brightfield', 'DAPI', 'GREEN', 'PE']

    for idx, mask in enumerate(masks):
        feature_dict = {'ImagePath': image_names[idx]}
        image = imgs[idx]

        apc, brightfield, dapi, gfp, pe = np.split(image, 5, axis=-1)
        channels = {
            'APC': np.squeeze(apc),
            'Brightfield': np.squeeze(brightfield),
            'DAPI': np.squeeze(dapi),
            'GREEN': np.squeeze(gfp),
            'PE': np.squeeze(pe)
        }

        cell_mask = keep_largest_object(mask)
        if cell_mask is None:
            continue

        labeled_mask, num_features = label(cell_mask)
        if num_features == 0:
            continue

        # Keep your existing summary features
        properties = regionprops_table(
            labeled_mask,
            properties=['area', 'eccentricity', 'major_axis_length']
        )

        feature_dict['Cell_Area'] = np.sum(properties['area'])
        feature_dict['Cell_Eccentricity'] = np.sum(properties['eccentricity'])
        feature_dict['Cell_MajorAxisLength'] = np.sum(properties['major_axis_length'])

        # Keep your old mean-intensity outputs as well
        for ch_name, ch_img in channels.items():
            intensity_props = regionprops_table(
                labeled_mask,
                intensity_image=ch_img,
                properties=['mean_intensity']
            )
            feature_dict[f'CellMeanIntensity_{ch_name}'] = np.mean(intensity_props['mean_intensity'])

        # Add requested Cells_Intensity_* features
        for ch_name in cell_intensity_channels:
            add_intensity_features(
                feature_dict=feature_dict,
                mask=cell_mask,
                channel=channels[ch_name],
                compartment='Cells',
                channel_name=ch_name
            )

        # Add requested Cells_Texture_* features
        for ch_name in texture_channels:
            add_texture_features(
                feature_dict=feature_dict,
                mask=cell_mask,
                channel=channels[ch_name],
                compartment='Cells',
                channel_name=ch_name,
                distance=6,
                levels=32
            )

        feature_df = pd.DataFrame([feature_dict])
        batch_feature_pd = pd.concat([batch_feature_pd, feature_df], ignore_index=True)

    return batch_feature_pd

def get_images(file_names):
    imgs = []
    image_5_channels = []
    valid_files = []

    for files in file_names:
        try:
            image = tiff.imread(files)
            image_5_channels.append(image)

            apc, brightfield, dapi, gfp, pe = np.split(image, 5, axis=-1)
            image_sum = apc + dapi + gfp + pe

            imgs.append(image_sum)
            valid_files.append(files)

        except Exception as e:
            print(f"Error processing {files}: {e}")

    return imgs, image_5_channels, valid_files

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
    imgs, image_5_channels, valid_files = get_images(filenames)

    if len(imgs) == 0:
        return pd.DataFrame()

    nucleus_masks = nucleus_model.eval(
        imgs, flow_threshold=0, diameter=None, resample=1, channels=[0, 0]
    )[0]

    masks = refine_mask(nucleus_masks, image_5_channels)
    features_df = extract_features(masks, image_5_channels, valid_files)
    return features_df

#####################
######### main ######
#####################

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file)])
    logger = logging.getLogger(__name__)

    image_info = build_image_info(input_folder)

    all_features = pd.DataFrame()
    image_paths = image_info['Image Path']
    length_pd = len(image_paths)

    logger.info("Starting image processing")

    with ThreadPoolExecutor(max_workers=10) as executor:
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
