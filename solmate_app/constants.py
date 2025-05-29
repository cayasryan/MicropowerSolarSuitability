LAND_LABELS = {
        10: "Tree Cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
        50: "Built-up", 60: "Bare/Sparse Veg", 70: "Snow/Ice", 80: "Water",
        90: "Wetlands", 95: "Mangroves", 100: "Moss & Lichen"
    }
UNSUITABLE_LC = {"Snow/Ice","Water","Wetlands","Mangroves","Cropland"}

THRESH = {
    "flood_extent": 0.4,
    "flood_duration": 10,
    "flood_mean": 1.0,
    "flood_max": 1.0,
    "fault_dist": 5,
    "res_dist": 10,
    "solar_min": 18000,
    "temp_low": 0,
    "temp_high": 45,
    "precip_high": 0.5,
}

GRID_TYPES_MAPPING = {
    "line": "Line",
    "substation": "Substation",
    "minor_line": "Minor Line",
}

RENAME_MAPPING = {
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'suitability': 'Suitability',
        'remarks': 'Remarks',
        'land_cover': 'Land Cover',
        'slope': 'Slope (degrees)',
        'elevation': 'Elevation (m)',
        'solar': 'Monthly Surface Solar Radiation (J/m²)',
        'temp': 'Mean 2m Temperature (°C)',
        'precip': 'Mean Monthly Precipitation (m)',
        'flood_occur': 'Annual Flood Occurrence',
        'flood_duration': 'Mean Flood Duration (days)',
        'flood_mean': 'Mean Flood Depth (m)',
        'flood_max': 'Max Flood Depth (m)',
        'fault_line_prox': 'Fault Line Proximity (m)',
        'residential_prox': 'Residential Area Proximity (m)',
        'airport_prox': 'Airport Proximity (m)',
        'road_prox': 'Main Road Proximity (m)',
        'protected_area_prox': 'Protected Area Proximity (m)',
        'grid_prox': 'Grid Proximity (m)',
        'nearest_protected_area': 'Nearest Protected Area',
        'nearest_grid': 'Nearest Grid Type',
        'spug_prox': 'SPUG Area Proximity (m)',
        'nearest_spug': 'Nearest SPUG Area',
        'in_protected_area': 'In Protected Area?',
        'in_spug': 'In SPUG Area?',
    }

HIGHLIGHT_COLORS = {
                'suitable': 'background-color: #d4edda',
                'likely_unsuitable': 'background-color: #f8d7da',
                'recommendation_suitable': 'background-color: #d4edda',
                'recommendation_unsuitable': 'background-color: #f8d7da',
                'highlight_affected_feature': 'background-color: #f8d7da',
                'summary_suitable': '#2e7d32',
                'summary_unsuitable': '#d32f2f',
                'issues_breakdown': '#E87461',
            }