import pandas as pd
import pickle
import glob
import os
import umap
from sklearn import datasets, decomposition
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.colors
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import umap.plot
from astropy.io import fits
from astropy import table
from astropy.table import Table
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
sns.set_style("white")
from sklearn.metrics import mean_squared_error

import bokeh.plotting as bpl
import bokeh.transform as btr
import holoviews as hv
import holoviews.operation.datashader as hd
from bokeh.models import Button  # for saving data
from bokeh.events import ButtonClick  # for saving data

from bokeh.layouts import column, row
from bokeh.models import CustomJS, TextInput, MultiLine
from bokeh.plotting import figure
import datashader as ds
from clustering_setup import plot_embedding

from warnings import warn

from umap.plot import _themes, _get_embedding, _to_hex

with fits.open(path+'/Data/catalogue_before_clustering_and_classification.fits') as data: #catalogue of sources to cluster 
    catalogue = table.Table(data[1].data)
    
for i in catalogue:
    name = i['Source_Name_1']
    names_list.append(name)
    
names_list = pd.DataFrame(names_list)

all_SB_data = pd.read_pickle(path +"/all_SB_data.pkl").T.reset_index(drop=True).astype(float) #pkl of surface brightness profiles, for each source to be clustered.

all_SB_data_scaled = ((all_SB_data.T- np.mean(all_SB_data.to_numpy(),axis=1))/np.std(all_SB_data.to_numpy(),axis=1)).T  #scale the SB profiles 
all_SB_data_scaled = all_SB_data_scaled.astype(np.float32).replace([np.inf, -np.inf], np.nan, inplace=False)
all_SB_data_scaled = all_SB_data_scaled.dropna()

all_host_data=pd.read_pickle(path +"/all_host_data.pkl").T.reset_index(drop=True).astype(float) #pkl file containing closest surface brightness spline point to host galaxy, for each source to be clustered.

all_data = pd.concat([all_SB_data_scaled,all_host_data,names_frame], axis=1, ignore_index=True)

mapper= umap.UMAP(n_neighbors=400, min_dist=1e-3, n_components=2, repulsion_strength=24) #change these parameters as needed
embedding = mapper.fit_transform(all_data.iloc[:,:-2])

#cluster and return associated labels. -1 = noisy
labels = hdbscan.HDBSCAN(
    min_samples=100,
    min_cluster_size=800,
    max_cluster_size=8500
).fit_predict(embedding)

clustered = (labels >= 0)
print((np.sum(clustered) / all_data.shape[0]*100), '% data clustered')
print(np.unqiue(labels), 'clusters identified')
#plot umap embedding, colour coded by HDBSCAN labels
p, ds, ds2=plot_embedding(
    mapper,
    labels=labels,
    hover_data=pd.DataFrame(
        np.array([labels, all_data.iloc[:,-1]]).T, index=all_data.index, columns=["label", "name"]
    ),
       
    point_size=2,
    tools=["lasso_select", "box_zoom", "reset", "save"], 
    interactive_sample_plot=True
)


bpl.show(p)
bpl.show(p)


    