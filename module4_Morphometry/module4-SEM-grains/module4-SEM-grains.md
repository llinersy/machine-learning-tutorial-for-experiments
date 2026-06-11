# Module 4: Image Processing & Morphometry
## **Objective:** Automatically extract physical descriptors (size, shape, count) from micrographs.

### 1. Setup and Artificial Micrograph Generation
#### - We simulate an SEM image with particles, noise, and uneven lighting (shadows).


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation, color

def generate_micrograph(n_particles=25, img_size=512):
    # 1. Create full 2D grids immediately to avoid dimension mismatch
    y_indices, x_indices = np.mgrid[:img_size, :img_size]
    
    # 2. Create the background (explicitly 512x512)
    # Using x_indices ensures img is shape (512, 512)
    img = 0.2 * (x_indices / img_size) 
    
    np.random.seed(42)
    for _ in range(n_particles):
        # Random parameters for grains
        r, c = np.random.randint(50, img_size-50, 2)
        radius_a = np.random.randint(15, 40)
        radius_b = np.random.randint(15, 40)
        angle = np.random.rand() * np.pi
        
        # 3. Create the mask using the same grids
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_rot = cos_a * (x_indices - c) + sin_a * (y_indices - r)
        y_rot = -sin_a * (x_indices - c) + cos_a * (y_indices - r)
        
        mask = (x_rot**2 / radius_a**2 + y_rot**2 / radius_b**2) <= 1
        
        # 4. Apply to image (safe because both are 512x512)
        img[mask] = np.random.uniform(0.6, 0.9)
        
    # 5. Add Gaussian noise
    img += np.random.normal(0, 0.05, img.shape)
    return np.clip(img, 0, 1)

# Generate and display
image = generate_micrograph()
plt.figure(figsize=(6,6))
plt.imshow(image, cmap='gray')
plt.title("Simulated SEM Micrograph (Grains with Shadows/Noise)")
plt.axis('off')
plt.show()
```


    
![png](output_2_0.png)
    


## 2. Beyond Simple Thresholding
#### Traditional Otsu thresholding often fails with uneven lighting. 
#### We use Local Thresholding and Watershed Segmentation.


```python
# Step 4: Distance Transform & Watershed (Separate touching particles)
from skimage.feature import peak_local_max

# 1. Calculate the distance to the background
distance = ndi.distance_transform_edt(cleaned)

# 2. Find peaks in the distance map (the centers of the grains)
# We use peak_local_max instead of local_maxima to allow for min_distance
coords = peak_local_max(distance, min_distance=20, labels=cleaned)

# 3. Create a mask of the peaks to use as markers for Watershed
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)

# 4. Run the Watershed algorithm
labels = segmentation.watershed(-distance, markers, mask=cleaned)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(distance, cmap='viridis')
plt.title("Distance Transform")
plt.subplot(1, 2, 2)
plt.imshow(color.label2rgb(labels, bg_label=0))
plt.title("Final Segmented Grains")
plt.show()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[12], line 9
          5 distance = ndi.distance_transform_edt(cleaned)
          7 # 2. Find peaks in the distance map (the centers of the grains)
          8 # We use peak_local_max instead of local_maxima to allow for min_distance
    ----> 9 coords = peak_local_max(distance, min_distance=20, labels=cleaned)
         11 # 3. Create a mask of the peaks to use as markers for Watershed
         12 mask = np.zeros(distance.shape, dtype=bool)


    File ~/Documents/Eventos/Organizacion_de_eventos/2026/Module4-Image_Processing/venv_lppi/lib/python3.11/site-packages/skimage/feature/peak.py:267, in peak_local_max(image, min_distance, threshold_abs, threshold_rel, exclude_border, num_peaks, footprint, labels, num_peaks_per_label, p_norm)
        262     coordinates = _get_high_intensity_peaks(
        263         image, mask, num_peaks, min_distance, p_norm
        264     )
        266 else:
    --> 267     _labels = _exclude_border(labels.astype(int, casting="safe"), border_width)
        269     if np.issubdtype(image.dtype, np.floating):
        270         bg_val = np.finfo(image.dtype).min


    TypeError: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'


## 3. Automated Feature Extraction & Morphometry
#### We calculate physical descriptors for every single grain found.


```python
props = measure.regionprops_table(labels, intensity_image=image, 
                                  properties=['label', 'area', 'equivalent_diameter', 
                                              'major_axis_length', 'minor_axis_length', 
                                              'orientation'])

df = pd.DataFrame(props)

# We filter out particles that are too small to have a physical 'width'
df = df[df['minor_axis_length'] > 0].copy()

# Now the formula is safe
df['aspect_ratio'] = df['major_axis_length'] / df['minor_axis_length']

# Final check: Remove any rows where aspect_ratio might still be non-finite
df = df[np.isfinite(df['aspect_ratio'])]

print(f"Total Particles Counted: {len(df)}")
print(df[['label', 'area', 'aspect_ratio']].head())

# Visualize the Distribution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(df['equivalent_diameter'], bins=10, color='skyblue', edgecolor='black')
plt.title("Grain Size Distribution")
plt.xlabel("Diameter (pixels)")

plt.subplot(1, 2, 2)
plt.hist(df['aspect_ratio'], bins=10, color='salmon', edgecolor='black')
plt.title("Shape Analysis (Aspect Ratio)")
plt.xlabel("Ratio (1.0 = Sphere)")
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[13], line 1
    ----> 1 props = measure.regionprops_table(labels, intensity_image=image, 
          2                                   properties=['label', 'area', 'equivalent_diameter', 
          3                                               'major_axis_length', 'minor_axis_length', 
          4                                               'orientation'])
          6 df = pd.DataFrame(props)
          8 # We filter out particles that are too small to have a physical 'width'


    NameError: name 'labels' is not defined


# 🧪 Module 4: Student Lab Challenges

### Task 1: Tuning the Denoising
The Gaussian filter `sigma=1` determines how much detail we blur. 
1. **Action:** Change `sigma` to **5**. 
2. **Observation:** What happens to the boundaries of the grains? Does the watershed still separate touching particles correctly?

### Task 2: Porosity Calculation
In membrane science, "Porosity" is the ratio of void space to total area.
1. **Action:** Calculate the porosity of this image.
   ```python
   total_pixels = image.size
   void_pixels = np.sum(cleaned == 0)
   porosity = (void_pixels / total_pixels) * 100
   print(f"Porosity: {porosity:.2f}%")


```python

```
