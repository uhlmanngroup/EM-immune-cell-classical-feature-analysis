import os, glob, base64
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.cluster import KMeans

from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, TapTool, CustomJS, Div
from bokeh.layouts import row, column, gridplot
from bokeh.transform import factor_cmap
from bokeh.palettes import Turbo256, Category10, Viridis256, Spectral11, Colorblind # Palettes for coloring
from pathlib import Path
from io import BytesIO

from PIL import Image

import umap

'''
This script generates UMAP projections of extracted morphological features coloured by different sets of labels.
The generated UMAPs are interactive and in the format of html.
'''

# ================= CONFIGURATION =================
DISPLAY_SIZE = (600, 600)
PLOT_SIZE = (500, 500)

def load_data_with_categories(filepaths):
    """
    Loads images and assigns categories.
    """
    # 1. Get files
    files = filepaths
    
    encoded_images = []
    categories = [] # New list for labels

    print(f"Processing {len(files)} images...")
    
    for i, f_path in enumerate(files):
        try:
            with Image.open(f_path) as img:
                img = img.convert('RGB')
                img.thumbnail(DISPLAY_SIZE)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                img_str = base64.b64encode(buffer.getvalue()).decode()
                encoded_images.append(f"data:image/jpeg;base64,{img_str}")
        except Exception as e:
            print(f"Error: {e}")
            
    return encoded_images

def plot_umap(filename = '', ref_file = None, type_idx_list = [], embedding_size = 2, n_neighbors=15, min_dist=0.1, names = [], labels = []):

    features = pd.read_csv(filename, index_col=0)
    features = features.dropna()
    OUTPUT_FILE = f'{filename.split(".")[0]}_all_plots.html'

    if ref_file:
        df = pd.read_csv(ref_file, index_col=[0])
        df = df.dropna()
    values_list = []
    classes_list = []

    if 'ROI' not in filename:
        features_data = features[
            [
                'volume_in_microns', 'extent', 'equiv_diameter', 'major_axis', 'minor_axis', 'surface_area', 'max_radius', 'sphericity',  'mean_intensity', 'st_dev', 'median_intensity', 'interquartile_range_intensity'\
                , 'total', 'haralick_1', 'haralick_2', 'haralick_3', 'haralick_4', 'haralick_5', 'haralick_6', 'haralick_7', 'haralick_8', 'haralick_9', 'haralick_10', 'haralick_11', 'haralick_12', 'haralick_13', \
                    # 'shape_edt_mean', 'shape_edt_std', 'shape_edt_median', 'shape_edt_interquartile_range'
            ]
        ].values
    else:
        features_data = features[
            [
                'volume_in_microns', 'extent', 'equiv_diameter', 'major_axis', 'minor_axis', 'max_radius', 'mean_intensity', 'st_dev', 'median_intensity', 'interquartile_range_intensity'\
                , 'total', 'haralick_1', 'haralick_2', 'haralick_3', 'haralick_4', 'haralick_5', 'haralick_6', 'haralick_7', 'haralick_8', 'haralick_9', 'haralick_10', 'haralick_11', 'haralick_12', 'haralick_13', \
                    'shape_edt_mean', 'shape_edt_std', 'shape_edt_median', 'shape_edt_interquartile_range'
            ]
        ].values        

    cell_names = features.index.tolist()
    img_names = [os.path.join('images', x.replace('tif', 'jpg')) for x in cell_names]
    b64_images = load_data_with_categories(img_names)

    # scaled_features_data = StandardScaler().fit_transform(features_data)
    # print(scaled_features_data.shape)

    # reducer_clustering = umap.UMAP(n_components=embedding_size, n_neighbors=30, min_dist=0.0,random_state=99) 
    reducer_visualization = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=99)
    # embedding_clustering = reducer_clustering.fit_transform(features_data) 
    embedding_visualization = reducer_visualization.fit_transform(features_data) 

    print('embedding size for visualization: ', embedding_visualization.shape)      

    for i, type_idx in enumerate(type_idx_list):

        if names[i] == 'cell type':        
            values = list(df['type']) #.map(type_to_idx)
            classes = list(type_idx.keys())
        elif names[i] == 'CD16 label':
            values = list(map(type_idx.get, list(df['CD16'])))
            classes = list(type_idx.values())
        elif names[i] == 'biopsy':
            values = list(map(type_idx.get, list(df['biopsy'])))
            classes = list(type_idx.values())
        elif names[i] == 'kmeans': 
            kmeans = KMeans(n_clusters=6, random_state=99, n_init="auto").fit(features_data)
            values = list(map(type_idx.get, kmeans.labels_))
            classes = list(type_idx.values())
        values_list.append(values)
        classes_list.append(classes)
        
    data = {
        'x' : embedding_visualization[:, 0],
        'y' : embedding_visualization[:, 1],
        'name' : img_names,
        'img' : b64_images,
        # DIFFERENT ATTRIBUTES FOR COLORING
        'cell_type_labels': values_list[0],
        'cd16_labels': values_list[1],
        'biopsy_labels': values_list[2],
        'kmeans_labels': values_list[3],
    }

    cat_names = ['cell_type_labels', 'cd16_labels', 'biopsy_labels', 'kmeans_labels',]

    ###########################################################################################
    # --- CRITICAL STEP: ONE SOURCE FOR ALL PLOTS ---
    source = ColumnDataSource(data=data)

    p_list = []

    for k in range(len(values_list)):
        # Select a palette based on how many classes you have
        if len(classes_list[k]) <= 10:
            palette = Colorblind[8] # Good distinct colors for few classes
        else:
            palette = Turbo256 # A massive rainbow gradient for many classes

        # Create the color mapper
        color_mapper = factor_cmap(
            field_name=cat_names[k], # The column in source to look at
            palette=palette, 
            factors=classes_list[k]
        )

        # --- PLOT WITH LEGEND ---
        p = figure(
            title=f"{labels[k]}) UMAP Projection with {names[k]} - {len(classes_list[k])} Categories",
            width=PLOT_SIZE[0], height=PLOT_SIZE[1],
            tools="tap,pan,wheel_zoom,reset",
            active_scroll="wheel_zoom"
        )

        # Draw circles
        # Note: 'legend_group' automatically groups them in the legend!
        p.scatter(
            'x', 'y', size=10, source=source,
            color=color_mapper,       # Use the mapper here
            legend_group=cat_names[k],  # Creates the legend automatically
            fill_alpha=0.9,
            # selection_color="black",  # Color when clicked
            selection_alpha=1.0,
            line_color=None,
            nonselection_fill_alpha=0.3,   # 0.0 is invisible, 1.0 is solid. 0.2 is "ghosted"
            nonselection_fill_color=color_mapper, # KEEP the original category color
            nonselection_line_color=None
        )
        p_list.append(p)


    for j, p in enumerate(p_list):
        p.legend.label_text_font_size = "8pt" 
        p.legend.title_text_font_size = "10pt"
        p.legend.title = cat_names[j]
        p.legend.location = "bottom_left"
        p.legend.click_policy = "hide" # BONUS: Click legend to toggle visibility!

    # --- 5. INTERACTION (Right Panel) ---
    div = Div(width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1], styles={'padding-left':'20px'}, text="""
        <div style="text-align: center; margin-top: 50px; color: #555; font-family: sans-serif;">
            <h3>Select an Image</h3>
            <p>Click a point to view details.</p>
        </div>
    """)

    callback = CustomJS(args=dict(source=source, div=div), code="""
        const indices = source.selected.indices;
        if (indices.length > 0) {
            const idx = indices[0];
            const img_data = source.data['img'][idx];
            const name = source.data['name'][idx];
            const cell_type = source.data['cell_type_labels'][idx];
            const cd16 = source.data['cd16_labels'][idx];
            const biopsy = source.data['biopsy_labels'][idx];
            const kmeans = source.data['kmeans_labels'][idx];
            
            div.text = `
                <div style="text-align: center; font-family: sans-serif;">
                    <h3 style="margin-bottom: 5px;">${name}</h3>
                    <span style="background-color: #eee; padding: 5px; border-radius: 4px;">
                        Cell type: <b>${cell_type}</b>
                        CD16 label: <b>${cd16}</b>
                        Biopsy: <b>${biopsy}</b>
                        Kmeans cluster: <b>${kmeans}</b>
                    </span>
                    <br><br>
                    <img src="${img_data}" style="max-width: 350px; border: 1px solid #ccc;">
                </div>
            `;
        }
    """)
    # Attach callback to the SHARED source
    source.selected.js_on_change('indices', callback)

    # ================= 5. FINAL LAYOUT =================
    # Create the 2x2 grid of plots
    grid = gridplot([
        [p_list[0], p_list[1]],
        [p_list[2], p_list[3]]
    ])

    # Put the Grid on the Left, Image Viewer on the Right
    layout = row(grid, div)

    output_file(OUTPUT_FILE)
    show(layout)



##################################################################################################################

if __name__ == '__main__':

    type_to_idx = {
        "PLT": 0,
        "Polymorph": 1,
        "Mononuclear_Small": 2,
        "Mononuclear_Large": 3,
        "Mononuclear_Very_large": 4,
        "Unknown": 5
    }

    idx_to_type_kmeans = {
        0 : 'KMeans cluster 0',
        1 : 'KMeans cluster 1',
        2 : 'KMeans cluster 2',
        3 : 'KMeans cluster 3',
        4 : 'KMeans cluster 4',
        5 : 'KMeans cluster 5'

    }

    idx_to_type_cd16 = {
        0 : 'CD16 Negative',
        1 : 'CD16 Positive',
        2 : 'CD16 Unknown (outside LM)'
    }

    idx_to_type_biopsy = {
        0 : 'EM04629',
        1 : 'EM04652',
        2 : 'EM04654',
    }

    embedding_size = 10

    type_idx_list = [type_to_idx, idx_to_type_cd16, idx_to_type_biopsy, idx_to_type_kmeans]
    names = ['cell type', 'CD16 label', 'biopsy', 'kmeans']
    labels = ['A', 'B', 'C', 'D']

    for filename in ['features_all.csv']:  # 'features_EM04629.csv', 'features_EM04652.csv', 'features_EM04654.csv', 
        # classes = np.unique(pd.read_csv(filename).type)
        print(f'plotting: {filename}')
        if '4629' in filename:
            plot_umap(filename=filename, type_idx=type_to_idx, n_neighbors=15, min_dist=0.0, show_image=True)
        elif '4652' in filename:
            plot_umap(filename=filename, type_idx=type_to_idx, n_neighbors=15, min_dist=0.0, show_image=True)
        elif '4654' in filename:
            plot_umap(filename=filename, type_idx=type_to_idx, n_neighbors=30, min_dist=0.5, show_image=True)
        else:
            plot_umap(filename=filename, ref_file = 'labels.csv', type_idx_list=type_idx_list, n_neighbors=45, min_dist=0.5, names=names, labels = labels)
