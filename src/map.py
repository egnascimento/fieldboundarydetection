import folium
import h3

filename = 'hexes.csv'


def save_to_map(hexes, pcts):

    center = h3.h3_to_geo_boundary(hexes[list(hexes.keys())[0]][0], geo_json=True)[0]
    center = [center[1], center[0]]
    map = folium.Map(location=center, zoom_start=17, max_zoom=24,
                     attr="Esri",
                     tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}")

    for key in hexes:
        #for hex in hexes[key]:
        for idx, hex in enumerate(hexes[key]):
            geometry = {"type": "Polygon", "coordinates": [h3.h3_to_geo_boundary(str(hex), geo_json=True)]}
            if pcts[key][idx] > 0.7:
                geo_j = folium.GeoJson(data=geometry, style_function=lambda x: {'fillColor': 'green', 'color': 'green', 'weight': '1'})
            else:
                geo_j = folium.GeoJson(data=geometry, style_function=lambda x: {'fillColor': 'yellow', 'color': 'yellow', 'weight': '1'})
            geo_j.add_child(folium.Popup(str(pcts[key][idx])))
            geo_j.add_to(map)

        border = []

        for h in hexes[key]:
            n = h3.k_ring(h, 2)
            print(h)
            print(n)

            for hex in n:
                if (hex not in hexes[key]) and (hex not in border):
                    border.append(hex)

        for hex in border:
            geometry = {"type": "Polygon", "coordinates": [h3.h3_to_geo_boundary(str(hex), geo_json=True)]}
            geo_j = folium.GeoJson(data=geometry, style_function=lambda x: {'fillColor': 'red', 'color': 'red', 'weight': '1'})
            #geo_j.add_child(folium.Popup(str(pcts[key][idx])))
            geo_j.add_to(map)

        map.save(key + '.html')
