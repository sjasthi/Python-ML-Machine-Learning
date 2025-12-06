# K-Means Clustering Images

This folder contains 8 educational visualizations for teaching K-Means Clustering to students.

## Image Descriptions

### 1. kmc_intro_analogy.png (277 KB)
**Purpose:** Introduction to K-Means concept
**Content:** 
- Before/After comparison showing fruits being organized
- Left side: All fruits mixed up (gray dots)
- Right side: Fruits grouped by size and sweetness into 3 colored clusters
- Shows centroids as yellow stars
**Use in lesson:** Perfect for explaining "What is K-Means?" section

### 2. kmc_party.png (489 KB)
**Purpose:** Step-by-step party analogy
**Content:** 
- Four panels showing the K-Means algorithm as a party
- Step 1: Random group leaders chosen
- Step 2: Everyone joins nearest leader
- Step 3: Leaders move to center of their groups
- Step 4: Final stable grouping
**Use in lesson:** Great for explaining "How K-Means Works: The Party Analogy"

### 3. kmc_steps.png (650 KB)
**Purpose:** Technical algorithm steps
**Content:**
- Four panels showing the formal K-Means process
- Step 1: Initialize K random centroids
- Step 2: Assign points to nearest centroid (with cluster circles)
- Step 3: Update centroids with arrows showing movement
- Step 4: Final converged state
**Use in lesson:** Use for technical explanation and code walkthrough

### 4. kmc_distance.png (149 KB)
**Purpose:** Distance calculation explanation
**Content:**
- Two points (A and B) on a 2D plane
- Right triangle showing Δx and Δy
- Visual representation of Euclidean distance formula
- Formula box: Distance = √((x₂ - x₁)² + (y₂ - y₁)²)
**Use in lesson:** Use when explaining distance measurement

### 5. kmc_k_values.png (377 KB)
**Purpose:** Comparison of different K values
**Content:**
- Three panels side by side
- K=2: Too few clusters (forced groupings)
- K=3: Just right (natural groupings) ✓
- K=5: Too many clusters (over-segmentation)
**Use in lesson:** Excellent for discussing how to choose K

### 6. kmc_color_quantization.png (102 KB)
**Purpose:** Real-world application - image compression
**Content:**
- Three images side by side
- Original: Full color gradient image
- K=16: Same image with only 16 colors
- K=4: Same image with only 4 colors
- Shows how K-Means reduces image complexity
**Use in lesson:** Perfect example of practical K-Means application

### 7. kmc_elbow_method.png (235 KB)
**Purpose:** Finding optimal K value
**Content:**
- Line graph showing K (x-axis) vs Inertia (y-axis)
- Red star marking the "elbow point" at K=3
- Vertical dashed line at optimal K
- Annotations explaining the elbow concept
- Shaded regions showing "sharp decrease" vs "diminishing returns"
**Use in lesson:** Essential for teaching the Elbow Method

### 8. kmc_wrong_k.png (928 KB)
**Purpose:** Comprehensive K value comparison
**Content:**
- Original data visualization showing 3 natural clusters
- Multiple panels showing K=2, 3, 4, 7, 10
- Visual demonstration of under-clustering and over-clustering
- Color-coded warnings (red X for bad, green ✓ for good)
- Summary box with tips for choosing K
**Use in lesson:** Great summary slide for K selection discussion

## Usage Tips

1. **Projection Order:** Follow the order of images as they appear in the markdown document
2. **Build Understanding:** Start with simple analogies (fruit, party) before technical details
3. **Interactive Discussion:** Use the "wrong K" images to have students identify problems
4. **Hands-on Practice:** After showing elbow method, have students create their own

## Technical Details

- All images: 300 DPI (high quality for projection)
- Format: PNG with transparency where applicable
- Color schemes: Designed to be colorblind-friendly
- Generated using: Python (matplotlib, scikit-learn, numpy)

## Customization

If you need to modify these images, the generation script is available:
`generate_kmc_images.py`

You can adjust:
- Colors and styling
- Data point sizes
- Number of data points
- K values shown
- Annotations and labels

---

Created for: Python Machine Learning Course
Instructor: Siva
Institution: Metro State University (via Learn and Help)
Date: December 2024
