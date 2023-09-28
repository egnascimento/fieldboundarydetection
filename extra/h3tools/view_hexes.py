import folium
import pandas as pd
import h3
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv



filename = 'hexes.csv'

with open(filename, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

print(data[0][0])

#print(df.to_numpy()[0])


center = h3.h3_to_geo_boundary(data[0][0], geo_json=True)[0]
center = [center[1], center[0]]
map = folium.Map(location=center,
                 zoom_start=17, max_zoom=24,
                 attr="Esri",
                 tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}")

# https://leafletjs.com/reference-1.6.0.html#path-option


for hex in data[0]:
    print(hex)
    geometry = {"type": "Polygon", "coordinates": [h3.h3_to_geo_boundary(str(hex), geo_json=True)]}

    geo_j = folium.GeoJson(data=geometry, style_function=lambda x: {'fillColor': 'black', 'color': 'black', 'weight': '1'})

    geo_j.add_to(map)


border = []

for h in data[0]:
    n = h3.k_ring(h)
    print(h)
    print(n)

    for hex in n:
        if (hex not in data[0]) and (hex not in border):
            border.append(hex)

for hex in border:
    geometry = {"type": "Polygon", "coordinates": [h3.h3_to_geo_boundary(str(hex), geo_json=True)]}
    geo_j = folium.GeoJson(data=geometry, style_function=lambda x: {'fillColor': 'yellow', 'color': 'green', 'weight': '1'})
    geo_j.add_child(folium.Popup(hex))

    geo_j.add_to(map)

map.save('hexes.html')
