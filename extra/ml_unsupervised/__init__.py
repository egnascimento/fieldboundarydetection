import folium
import pandas as pd
import h3
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

filename = 'data/hex_index_L3=832600fffffffff.parquet'

# select method: can be dbscan or kmeans
method = 'kmeans'

# if kmeans, configure the number of clusters
clusters = 4

# if dbscan, configure the metric and the respective distance
metric = 'cosine'
eps = 0.0003

s2b_df = pd.read_parquet(filename, engine='pyarrow')

print(s2b_df.shape)
s2b_df = s2b_df.drop_duplicates(subset=['hex'], keep='first')
print(s2b_df.shape)

s2b_bands_df = s2b_df[['B01','B02','B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B8A', 'B11', 'B12']]

if method == 'kmeans':
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(s2b_bands_df)
    pca = PCA(n_components=2)
    pca_array = pca.fit_transform(s2b_bands_df)
    pca_array = np.concatenate((pca_array, np.array([kmeans.labels_]).T), axis=1 )
    pca_df = pd.DataFrame(pca_array, columns=['pca1', 'pca2', 'y'])
    sns.jointplot(data=pca_df, x="pca1", y="pca2", hue='y')
    plt.show()
    y = kmeans.labels_
    filename = 'output/map-' + filename + '-' + method + '-' + str(clusters) + '.html'

elif method == 'dbscan':
    metric = 'cosine'
    eps = 0.0003

    db = DBSCAN(metric=metric, eps=eps, min_samples=100).fit(s2b_bands_df)
    print('Core samples', db.core_sample_indices_, len(db.core_sample_indices_))

    pca = PCA(n_components=2)
    pca_array = pca.fit_transform(s2b_bands_df)
    pca_array = np.concatenate((pca_array, np.array([db.labels_]).T), axis=1 )
    pca_df = pd.DataFrame(pca_array, columns=['pca1', 'pca2', 'y'])
    sns.jointplot(data=pca_df, x="pca1", y="pca2", hue='y')
    plt.show()
    y = db.labels_
    filename = 'output/map-' + filename + '-' + method + '-' + metric + '-' + str(eps).replace('.', '_') + '.html'

    for n in np.unique(np.array(db.labels_)):
        print(str(n) + ' : ' + str(sum(db.labels_ == n)))

s2b_df = s2b_df.reset_index()

color_palette = ["#ff0000","#5eba41","#004000","#a9b53d","#d46ad3","#66b663","#d24290","#57c197","#ce4d31","#48bbd2","#d59632",
"#6687c7","#5a8032","#d78fc9","#36815b","#c54e67","#c5a767","#965089","#826a29","#ca765c", "#ffffff"]

s2b_bands_labeled_df = pd.concat([s2b_df,pd.DataFrame(y, columns=['field'])], axis=1)
center = h3.h3_to_geo_boundary(s2b_bands_labeled_df.loc[s2b_bands_labeled_df.index[0], 'hex'], geo_json=True)[0]
center = [center[1], center[0]]
map = folium.Map(location=center,
                 zoom_start=17, max_zoom=24,
                 attr="Esri",
                 tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}")

# https://leafletjs.com/reference-1.6.0.html#path-option

for index, row in s2b_bands_labeled_df.iterrows():
    geometry = {"type": "Polygon", "coordinates": [h3.h3_to_geo_boundary(row['hex'], geo_json=True)]}

    if row['field'] < 20:
        geo_j = eval("folium.GeoJson(data=geometry, style_function=lambda x: {'fillColor': '" + color_palette[row['field']+1] + "', 'color': '" + color_palette[row['field']+1] + "', 'weight': '1'})")
    else:
        geo_j = folium.GeoJson(data=geometry, style_function=lambda x: {'fillColor': 'black', 'color': 'black', 'weight': '1'})

    geo_j.add_child(folium.Popup(str(row['field']) ))
    geo_j.add_to(map)

map.save(filename)
