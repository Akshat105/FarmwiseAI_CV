# FarmwiseAI_CV
# Tea Plantation Analysis Using Geospatial Data

This project aims to analyze high-resolution geo-referenced imagery of a tea plantation, captured via drone, for the purpose of automating tea leaf harvesting.

## Objective

The primary goal of this project is to:

- Classify vegetation regions within the tea plantation as polygons.
- Calculate the area of each classified polygon.
- Determine the width of plantation rows and identify regions where the bush width deviates from the standard range.
- Generate navigation aid data for an automated tea rover.

## Input Data

### Highly accurate data:

- **GeoTIFF Image:** High-resolution drone imagery of the tea plantation.

### Less accurate data (photogrammetry points):

- **Digital Terrain Model (DTM):** Elevation data representing the bare ground surface.
- **Digital Surface Model (DSM):** Elevation data including trees, buildings, and other objects.
- **Digital Elevation Model (DEM):** A 3D representation of the terrain's surface.

## Expected Output

- **Vegetation Polygons (.shp format):** Classified vegetation regions in the shapefile format.
- **Area Estimation:** Attribute dataset containing the area of each classified polygon.
- **Width Analysis Report:** Detailed report highlighting the width of each plantation row.
- **Navigation Map for Tea Rover:** A map (latitude-longitude path) to assist the tea rover in navigating the plantation.

## Setup and Execution

1. Ensure you have the required Python libraries installed:

   ```bash
   pip install geopandas rasterio shapely matplotlib
