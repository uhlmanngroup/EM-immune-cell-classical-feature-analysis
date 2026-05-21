import os
import numpy as np
# import vigra
import pandas as pd
import tifffile as tf

from mahotas.features import haralick

from skimage import filters
from skimage.morphology import binary_opening, binary_closing, ball, binary_erosion, binary_dilation, dilation, erosion, label, remove_small_objects
from skimage.segmentation import watershed
from skimage.measure import regionprops, marching_cubes, mesh_surface_area

from scipy.ndimage import binary_fill_holes, distance_transform_edt

'''
This script calculates classical morphological features given EM images and corresponding segmentation masks, and save the results as a csv file.
Part of the code is adapted from https://github.com/mobie/platybrowser-project/blob/main/mmpb/extension/attributes/morphology_impl.py
'''

# DIR for image and mask folders (containing tif files)
imageDir = 'tif'
maskDir = 'mask'

outputDir = 'features'
os.makedirs(outputDir, exist_ok=True)

features_all = []
index = []

for filename in os.listdir(imageDir):
    if os.path.exists(os.path.join(maskDir, filename)):
        image = tf.imread(os.path.join(imageDir, filename))
        print(f"image: {filename} loaded!")
        index += [filename]

        mask = tf.imread(os.path.join(maskDir, filename)) #.astype(np.uint8)
        # print(np.max(mask), np.min(mask))
        print(f"mask: {filename} loaded")

        # set spacing depending on the images
        spacing = np.array([0.01214, 0.01214, 0.05]) if '4652' in filename else np.array([ 0.0100024, 0.0100024, 0.05])
        print('spacing: ', spacing)

        mask = binary_fill_holes(mask).astype(np.uint8)

        ##### calculate morphology features #####
        morph_features = regionprops(mask, image)

        volume_in_pix = morph_features[0]['area']
        volume_in_microns = np.prod(spacing) * volume_in_pix
        extent = morph_features[0]['extent']
        equiv_diameter = morph_features[0]['equivalent_diameter_area']
        major_axis = morph_features[0]['axis_major_length']
        minor_axis = morph_features[0]['axis_minor_length']

        verts, faces, normals, values = marching_cubes(mask, spacing=tuple(spacing), level=0.5)
        surface_area = mesh_surface_area(verts, faces)

        sphericity = (36 * np.pi * (float(volume_in_microns) ** 2)) / (float(surface_area) ** 3)

        edt = distance_transform_edt(mask, sampling=spacing, return_distances=True)
        max_radius = np.max(edt)

        print('morphology features: ', volume_in_microns, extent, equiv_diameter, major_axis, minor_axis, surface_area, max_radius, sphericity)

        ##### calculate intensity features #####
        intensity_vals_in_mask = image[mask==1]

        mean_intensity = np.mean(intensity_vals_in_mask, dtype=np.float64)
        st_dev = np.std(intensity_vals_in_mask, dtype=np.float64)
        median_intensity = np.median(intensity_vals_in_mask)

        quartile_75, quartile_25 = np.percentile(intensity_vals_in_mask, [75, 25])
        interquartile_range_intensity = quartile_75 - quartile_25

        total = np.sum(intensity_vals_in_mask, dtype=np.float64)

        print('intensity features: ', mean_intensity, st_dev, median_intensity, interquartile_range_intensity, total)

        ##### calculate texture features
        image_copy = image.copy()
        image_copy[mask==0] = 0

        try:
            hara = haralick(image_copy, ignore_zeros=True, return_mean=True, distance=2)

        except ValueError:
            print('Texture computation failed - can happen when using ignore_zeros')
            hara = (0.,) * 13

        print('haralick textures: ', hara)

        edt_norm = edt / np.max(edt)
        shape_edt = edt_norm[mask==1]

        shape_edt_mean = np.mean(shape_edt, dtype=np.float64)
        shape_edt_std = np.std(shape_edt, dtype=np.float64)
        shape_edt_median = np.median(shape_edt)

        shape_edt_quartile_75, shape_edt_quartile_25 = np.percentile(shape_edt, [75, 25])
        shape_edt_interquartile_range = shape_edt_quartile_75 - shape_edt_quartile_25

        print('distance map intensity features: ', shape_edt_mean, shape_edt_std, shape_edt_median, shape_edt_interquartile_range)
        
        features = list([volume_in_microns, extent, equiv_diameter, major_axis, minor_axis, surface_area, sphericity, max_radius, mean_intensity, st_dev, median_intensity, interquartile_range_intensity, total])
        features += list(hara)
        features += list([shape_edt_mean, shape_edt_std, shape_edt_median, shape_edt_interquartile_range])

        features_all.append(features)

columns = ['volume_in_microns', 'extent', 'equiv_diameter', 'major_axis', 'minor_axis', 'surface_area', 'max_radius', 'sphericity',  'mean_intensity', 'st_dev', 'median_intensity', 'interquartile_range_intensity'\
           , 'total', 'haralick_1', 'haralick_2', 'haralick_3', 'haralick_4', 'haralick_5', 'haralick_6', 'haralick_7', 'haralick_8', 'haralick_9', 'haralick_10', 'haralick_11', 'haralick_12', 'haralick_13', \
            'shape_edt_mean', 'shape_edt_std', 'shape_edt_median', 'shape_edt_interquartile_range']
df = pd.DataFrame(np.array(features_all), columns=columns, index=index)
csv_filename = 'features.csv'
df.to_csv(csv_filename)
