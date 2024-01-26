#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install geopandas


# In[2]:


pip install pandas shapely fiona descartes


# In[3]:


pip install rasterio


# In[4]:


pip install rasterio matplotlib


# In[ ]:


# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.plot import show
from skimage import segmentation


# In[ ]:


import rasterio
from rasterio.plot import show


# In[ ]:


def load_drone_image(file_name):
    """
    Load a drone image from the specified file path using rasterio.
    
    Parameters:
    - file_path (str): Path to the drone image file.
    
    Returns:
    - drone_image (numpy.ndarray): The loaded drone image as a NumPy array.
    """
    # Open the drone image using rasterio
    with rasterio.open(file_name) as src:
        # Read the image data
        drone_image = src.read()
    
    return drone_image


# In[ ]:


from rasterio.enums import Resampling

def load_drone_image(file_path, transform=None):
    with rasterio.open(file_path) as src:
  
        drone_image = src.read()
        
        if transform:
            src.transform = transform

    return drone_image


# In[ ]:


# Load high-resolution drone imagery
drone_image = load_drone_image('image.tif')


# In[10]:


#View Image
def visualize_rgb_image(file_path):
    with rasterio.open(file_path) as src:
        # Read the Red, Green, and Blue bands
        red_band = src.read(1)
        green_band = src.read(2)
        blue_band = src.read(3)

        # Stack the bands to create an RGB image
        rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)

        # Display the RGB image
        plt.imshow(rgb_image)
        plt.title('RGB Image')
        plt.show()

image_path = 'img.tif'
visualize_rgb_image(image_path)


# In[2]:



def visualize_image_with_dtm(image_path, dtm_path):
    # Load the image
    with rasterio.open(image_path) as img_src:
        image = img_src.read()

    # Load the DTM
    with rasterio.open(dtm_path) as dtm_src:
        dtm = dtm_src.read(1)

    # Visualize the DTM
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(dtm, cmap='terrain', extent=rasterio.plot.plotting_extent(dtm_src))
    plt.title('Digital Terrain Model (DTM)')
    plt.colorbar(label='Elevation (meters)')

    # Determine the number of subplots based on the number of bands in the image
    num_subplots = image.shape[0] if image.ndim == 3 else 1

    # Visualize individual bands of the image
    for i in range(num_subplots):
        plt.subplot(1, num_subplots + 1, i + 2)
        plt.imshow(image[i, :, :], cmap='gray', extent=rasterio.plot.plotting_extent(img_src))
        plt.title(f'Band {i + 1}')

    plt.tight_layout()
    plt.show()

image_path = 'image.tif'
dtm_path = 'dsm.tif'
visualize_image_with_dtm(image_path, dtm_path)


# In[12]:


def visualize_drone_image_with_dsm(drone_image_path, dsm_path):
    # Load the drone image
    with rasterio.open(drone_image_path) as drone_src:
        drone_image = drone_src.read()

    # Load the DSM
    with rasterio.open(dsm_path) as dsm_src:
        dsm = dsm_src.read(1)

    # Visualize the drone image and DSM together
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # Assuming the drone image is a true-color image
    axs[0].imshow(np.clip(drone_image / 255.0, 0, 1).transpose(1, 2, 0), extent=rasterio.plot.plotting_extent(drone_src))
    axs[0].set_title('High-Resolution Drone Image')

    im = axs[1].imshow(dsm, cmap='terrain', extent=rasterio.plot.plotting_extent(dsm_src))
    axs[1].set_title('Digital Surface Model (DSM)')

    # Add colorbar
    cbar = axs[1].figure.colorbar(im, ax=axs[1], label='Elevation (meters)')

    plt.show()


drone_image_path = 'image.tif'
dsm_path = 'dsm.tif'
visualize_drone_image_with_dsm(drone_image_path, dsm_path)


# In[13]:


# Load the drone image
with rasterio.open('image.tif') as drone_src:
    drone_image = drone_src.read()

# Load the orthomosaic
with rasterio.open('orthomosaic/ORTHO.tif') as ortho_src:
    ortho_image = ortho_src.read()

# Visualize the drone image and orthomosaic together
fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

# Assuming the drone image is a true-color image
axs[0].imshow(np.clip(drone_image / 255.0, 0, 1).transpose(1, 2, 0), extent=rasterio.plot.plotting_extent(drone_src))
axs[0].set_title('High-Resolution Drone Image')

# Assuming the orthomosaic is a true-color image
axs[1].imshow(np.clip(ortho_image / 255.0, 0, 1).transpose(1, 2, 0), extent=rasterio.plot.plotting_extent(ortho_src))
axs[1].set_title('Orthomosaic')

plt.show()


# In[14]:


import rasterio
import matplotlib.pyplot as plt

# Load the image
with rasterio.open('image.tif') as image_src:
    image = image_src.read()

# Load the DEM
with rasterio.open('dem.tif') as dem_src:
    dem = dem_src.read(1)  # Assuming it's a single-band DEM

# Perform some basic visualization (you may need to adjust depending on your data)
fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

axs[0].imshow(np.clip(image / 255.0, 0, 1).transpose(1, 2, 0), extent=rasterio.plot.plotting_extent(image_src))
axs[0].set_title('Original Image')

axs[1].imshow(dem, cmap='terrain', extent=rasterio.plot.plotting_extent(dem_src))
axs[1].set_title('Digital Elevation Model (DEM)')

plt.show()


# In[29]:


pip install opencv-python


# In[30]:


pip install opencv-contrib-python


# In[76]:


#Classifica7on of Vegeta7on
import cv2
import numpy as np

# Load the image (replace 'path/to/your/image.tif' with the actual path)
image = cv2.imread("image.tif")

# Convert to grayscale (optional, might help with edge detection)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection (adjust thresholds if needed)
edges = cv2.Canny(gray, 100, 200)  # Adjust thresholds for sensitivity

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Select the blue line contour based on color (HSV thresholding)
blue_lower = np.array([100, 50, 50], dtype="uint8")  # Adjust ranges as needed
blue_upper = np.array([140, 255, 255], dtype="uint8")
mask = cv2.inRange(image, blue_lower, blue_upper)
blue_contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Choose the largest blue contour (assuming line is largest blue object)
if len(blue_contour) > 0:
    blue_outline_contour = max(blue_contour, key=cv2.contourArea)

# Draw the outline on a copy of the image or create a new image
result = image.copy()  # Draw on original image
# result = np.zeros_like(image)  # Create a new blank image

cv2.drawContours(result, [blue_outline_contour], -1, (255, 0, 0), 2)  # Blue outline

# Display or save the result
cv2.imshow("Vegetation_polygon", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imwrite("vegetation_polygon.jpg", result)


# In[ ]:


#area Estimation
# Load vegetation polygons
vegetation_polygons = gpd.read_file('output/vegetation_polygons.shp')

# Calculate area for each polygon
vegetation_polygons['area'] = vegetation_polygons.geometry.area

# Save attribute dataset
vegetation_polygons.to_file('output/area_estimation_dataset.shp')


# In[ ]:


#Width Analysis
# Load vegetation polygons
vegetation_polygons = gpd.read_file('output/vegetation_polygons.shp')

# Calculate area for each polygon
vegetation_polygons['area'] = vegetation_polygons.geometry.area

# Save attribute dataset
vegetation_polygons.to_file('output/area_estimation_dataset.shp')
Step 3: Width Analysis
python
Copy code
import numpy as np

# Load vegetation polygons
vegetation_polygons = gpd.read_file('output/vegetation_polygons.shp')

# Calculate width for each row
def calculate_row_widths(polygons):
    # Implement your row width calculation logic here
    # Example: Calculate the distance between rows based on centroids
    row_widths = []
    for i, row in polygons.iterrows():
        centroid = row.geometry.centroid
        # Implement logic to calculate distance between centroids of adjacent rows
        # Append the result to row_widths
    return row_widths

row_widths = calculate_row_widths(vegetation_polygons)

# Identify regions with width deviation
def identify_deviation_regions(widths, threshold):
    # Implement your logic to identify regions with width deviation
    # Example: Find regions where width is above/below a certain threshold
    deviation_regions = np.where(np.array(widths) > threshold)[0]
    return deviation_regions

deviation_regions = identify_deviation_regions(row_widths, threshold=1.2)

# Save width analysis report
with open('output/width_analysis_report.txt', 'w') as report_file:
    report_file.write("Deviation Regions: {}".format(deviation_regions))


# In[ ]:


# Navigation Aid for Tea Rover
# Load vegetation polygons
vegetation_polygons = gpd.read_file('output/vegetation_polygons.shp')

# Generate navigation map for tea rover
def generate_navigation_map(polygons):
    # Implement your logic to generate a navigation map
    # Example: Use polygon centroids as waypoints for the rover
    navigation_map = polygons.geometry.centroid
    return navigation_map

navigation_map = generate_navigation_map(vegetation_polygons)

# Save navigation map as a path in lat-long format
navigation_map.to_file('output/navigation_map.geojson', driver='GeoJSON')

