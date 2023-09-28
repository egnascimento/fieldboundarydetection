import pandas as pd
from pathlib import Path
import folium
import h3
from glob import glob

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None

pfolders = glob('D:\\173_seeding_harvest_joined_USCA.parquet\\positive_samples\\*')
nfolders = glob('D:\\173_seeding_harvest_joined_USCA.parquet\\negative_samples\\*')

for f in pfolders:
    L3hex = f.split('=')[1]

    pdir = Path(f)
    ndir = Path(f.replace('positive_samples', 'negative_samples'))

    try:
        pdf = pd.concat(
            pd.read_parquet(parquet_file)
            for parquet_file in pdir.glob('*.parquet')
        )

        ndf = pd.concat(
            pd.read_parquet(parquet_file)
            for parquet_file in ndir.glob('*.parquet')
        )

        print(pdf.columns)
        print(pdf.dtypes)
        print(pdf['scene_id'].unique())

        print('Unique L12 hexes:', len(pdf['hex'].unique()))

        for scene in pdf['scene_id'].unique():
            print(scene, sum(pdf['scene_id']==scene))

        fields = pdf['FIELD_OPERATION_GUID'].unique()
        #print(fields)

        fields = ndf['FIELD_OPERATION_GUID'].unique()
        #print(fields)

        scene_ids = pdf['scene_id'].unique()
        #print(scene_ids)

        print(pdf.shape)
        print(ndf.shape)
        pdf.drop_duplicates(subset=["hex"], keep='first', inplace=True)
        ndf.drop_duplicates(subset=["hex"], keep='first', inplace=True)
        print(pdf.shape)
        print(ndf.shape)

        pdf_hexes = pdf['hex'].unique()
        ndf_hexes = ndf['hex'].unique()
        common_hexes_pn = set(pdf_hexes).intersection(ndf_hexes)
        #print(len(common_hexes_pn))
        common_hexes_np = set(ndf_hexes).intersection(pdf_hexes)
        #print(len(common_hexes_np))

        print(len(set(common_hexes_pn).intersection(common_hexes_np)))

        amb = round((len(common_hexes_np) / (pdf.shape[0])) * 100)
        print('Percentage of ambiguous negative samples:', amb)

        fields = pdf['FIELD_OPERATION_GUID'].unique()
        #print(fields)



        #print(pdf.shape, ndf.shape)


        center = h3.h3_to_geo_boundary(L3hex, geo_json=True)[0]
        center = [center[1], center[0]]

        map = folium.Map(location=center, zoom_start=17, max_zoom=24,
                         attr="Esri",
                         tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}")

        geometry = {"type": "Polygon", "coordinates": [h3.h3_to_geo_boundary(L3hex, geo_json=True)]}
        geo_j = folium.GeoJson(data=geometry,
                               style_function=lambda x: {'fillColor': 'white', 'color': 'white', 'fill_opacity': '0.1', 'weight': '2'})
        geo_j.add_to(map)

        for index, row in pdf.iterrows():
            if row['hex'] not in common_hexes_pn:
                geometry = {"type": "Polygon", "coordinates": [h3.h3_to_geo_boundary(str(row['hex']), geo_json=True)]}
                geo_j = folium.GeoJson(data=geometry, style_function=lambda x: {'fillColor': 'green', 'color': 'green', 'weight': '1'})
                geo_j.add_child(folium.Popup(str(row['FIELD_OPERATION_GUID'])))
                geo_j.add_to(map)

        for index, row in ndf.iterrows():
            if row['hex'] not in common_hexes_pn:
                geometry = {"type": "Polygon", "coordinates": [h3.h3_to_geo_boundary(str(row['hex']), geo_json=True)]}
                geo_j = folium.GeoJson(data=geometry, style_function=lambda x: {'fillColor': 'red', 'color': 'red', 'weight': '1'})
                geo_j.add_child(folium.Popup(str(row['FIELD_OPERATION_GUID'])))
                geo_j.add_to(map)

        for hex in common_hexes_pn:
            geometry = {"type": "Polygon", "coordinates": [h3.h3_to_geo_boundary(str(hex), geo_json=True)]}
            geo_j = folium.GeoJson(data=geometry,
                                   style_function=lambda x: {'fillColor': 'magenta', 'color': 'magenta', 'weight': '1'})
            geo_j.add_to(map)

        map.save(L3hex + '_' + str(amb) + '.html')
    except Exception as e:
        print("Issue with L3 Hex: ", L3hex, str(e))