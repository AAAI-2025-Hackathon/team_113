
# Check-In

- Title of your submission: **Detecting Rain Induced Landslides** (Theme: AI for Social Good)
- Team Members: Shelly Gupta, Hardik Sharma
- [x]  All team members agree to abide by the Hackathon Rules
- [x]  This AAAI 2025 hackathon entry was created by the team during the period of the hackathon, February 17 – February 24, 2025
- [x]  The entry includes a 2-minute maximum length demo video here: [Link](https://tuprd-my.sharepoint.com/:v:/g/personal/tuk40762_temple_edu/EelYJbrcfqZCv0-md6BFYzoB5Ua6hw9aWzsbsaAPMt1Zhg?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=vvlgXb)
- [x]  The entry clearly identifies the selected theme in the README and the video.


**If anybody wants to recreate our results, please download the Dataset ([Link](https://tuprd-my.sharepoint.com/:u:/g/personal/tur23692_temple_edu/EaGg9Go8cDhOnD6Aoc3B5SIBOiyeQNKlbSfxFvha1Kxszw?e=n3ZZD4)) and the necessary Python codes from the folder *Final_Modeling***
## Problem Statement

Landslides pose a significant threat in Oregon, affecting thousands of residents annually due to the state's unstable geology and frequent torrential rains. Oregon experiences more landslides than other mountainous regions, primarily triggered by rainfall rather than seismic activity. This makes it an ideal location for studying rain-induced landslides, as the state contains some of the most landslide-prone counties in the United States. **Our project focuses on detecting rain-induced landslides in a section of Oregon (left = -124.0006, bottom = 41.9994, right = -122.9994, and top = 43.0006) by analyzing various atmospheric and various soil attributes**. We utilize machine learning techniques to process this data and generate daily landslide probability signals for locations on a grid. This approach aims to improve landslide prediction accuracy, potentially reducing the risks associated with these natural disasters. By developing a more robust early warning system, we can help mitigate personal losses and material damages caused by landslides in Oregon and potentially extend this methodology to other vulnerable regions.
## Overview
Our approach to detecting rain-induced landslides in Oregon utilizes a comprehensive set of environmental and geological data sources. We incorporate matrices from **NLCD 2021 vegetation data**, providing detailed land cover information crucial for understanding surface conditions. **Soil composition data from SOLUS100** offers insights into soil properties essential for assessing slope stability. **USGS elevation data** contributes critical topographical information for identifying areas with steep slopes and complex terrain prone to landslides. **ERA5 soil attributes data** provides valuable information on soil moisture and other properties affecting landslide risk. To detect and validate landslide occurrences, we employ **NASA Global Landslide Nowcast data** as our ground truth.  We process geoimages to extract and structure relevant spatial information, preparing matrices that represent environmental attributes across the predefined grid. These processed matrices serve as input for our **multimodal Fully Convolutional Networks (FCNs) / U-Net**, which predicts the corresponding landslide probability for each grid cell. We train on year 2018 and 2019 and test on year 2020. By combining these diverse datasets and leveraging advanced machine learning techniques, our method captures the complex interactions between vegetation, soil, topography, and atmospheric conditions, enhancing the accuracy of landslide prediction in Oregon’s vulnerable terrain.

## Dataset
The datasets used for this project can be found on the following links:

- [NLCD 2021 vegetation data](https://www.mrlc.gov/data/nlcd-2021-land-cover-conus)
- [Soil composition data from SOLUS100](https://agdatacommons.nal.usda.gov/articles/dataset/Data_from_Soil_Landscapes_of_the_United_States_100-meter_SOLUS100_soil_property_maps_project_repository/25033856)
- [USGS elevation data](https://apps.nationalmap.gov/downloader/)
- [ERA5 soil attributes data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview)
- [NASA Global Landslide Nowcast data](https://catalog.data.gov/dataset/global-landslide-nowcast-from-lhasa-l4-1-day-1-km-x-1-km-version-1-1-global-landslide-nowc-e0c8f)


To construct our dataset, we collected data from six key resources, filtering them within the geographic bounds defined by the coordinates: left = -124.0006, bottom = 41.9994, right = -122.9994, and top = 43.0006. This spatial selection ensures that our analysis focuses on a well-defined region within Oregon. Among these datasets, ERA5 provides temporal soil attribute data, capturing daily variations in crucial factors such as soil moisture and precipitation, and NASA Global Landslide Nowcast provides daily landslides labels in form of a grid. In contrast, the remaining datasets (NLCD 2021 vegetation data, SOLUS100 soil composition data, USGS elevation data) are static, offering essential structural and topographical insights. 

Shape and type of the above data is as follows:

- **Elevation Data** : (10812, 10812), continuous
- **Vegetation Data** : (15, 4353, 1547), categorical    {where 15 are the different types of vegetation}
- **Soil Composition Data** : (18, 7, 1306, 464), continuous {where 18 are the different types of soil composition and 7 are the different depth measures (in cm)}
- **Soil Variable Data** : (1096, 28, 5, 5), continuous {where 1096 are the days and 28 are the different soil variables}
- **NASA Global Landslide Nowcast Data** : (1096, 120, 29640), multi-class label, {0 = No landslide, 1 = Minor intensity Landslide, 2 = Major intensity landslide}



## Modeling

Given the high dimensionality of our dataset, it is essential to reduce the size of each feature matrix and the output label matrix while preserving critical spatial and temporal information. To achieve this, we apply 2 different model approaches.

- **Approach 1** : We apply the following preprocessing steps to resize our input and output matrices:
  - **Vegetation Data**: Since vegetation data is categorical, we use the F.interpolate() function with mode='nearest' to resize it to a shape of (15, 120, 120), ensuring that class labels remain intact.
  - **Elevation Data**: As elevation is a continuous variable, we apply F.interpolate() with mode='bilinear' to resize it to (1, 120, 120), maintaining smooth transitions in topographical variations.
  - **Soil Variable Data**: This dataset contains temporal soil attributes and is resized to (1096, 28, 120, 120) using F.interpolate() with mode='bilinear' to preserve continuous data variations across time and space.
  - **Soil Composition Data**: Initially, the soil composition data is reshaped into (18, 1306, 464) to derive meaningful representations for each of the 18 soil types across 7 depth levels. The transformed matrix is then resized to (18, 120, 120) using F.interpolate() with mode='bilinear', ensuring spatial consistency while maintaining the continuous nature of the data.
  - **NASA Global Landslide Labels**: The landslide label matrix is resized to (1096, 120, 120) using a custom function. The original 29,640 columns are divided into 120 chunks, each containing 247 columns, with the maximum label within each chunk assigned to represent that particular grid cell.

These preprocessing steps effectively reduce computational complexity while ensuring that the spatial and temporal patterns necessary for landslide prediction remain well-preserved.


- **Approach 2** : We apply the following preprocessing steps to resize our input matrices:  

  - **Vegetation Data**: VegetationNet() is designed to process vegetation data with an input size of (batch, 15, 4353, 1547), producing a standardized output of (batch, 15, 120, 120). The model consists of two convolutional layers that downsample the input while extracting relevant vegetation features. The first 7×7 convolution with stride=4 expands the feature maps from 15 to 32, followed by a 3×3 convolution with stride=2 that maps the features back to 15 channels. After feature extraction, adaptive average pooling ensures that all vegetation matrices conform to a (120, 120) grid.

  - **Elevation Data**: ElevationNet() processes high-resolution elevation data with an input size of (batch, 1, 10812, 10812), reducing it to a standardized output of (batch, 1, 120, 120). The model consists of a series of convolutional layers that progressively downsample the spatial dimensions while preserving elevation features. The first convolution layer applies a 7×7 kernel with stride=4 to rapidly reduce resolution, followed by a 3×3 kernel with stride=2 that further refines feature extraction. The final 3×3 convolution restores a single-channel representation before applying adaptive average pooling, which ensures a fixed output resolution of (120, 120). 

  - **Soil Variable Data**: SoilVariableNet() processes soil attribute data, where the input size is exceptionally small, (batch, 28, 5, 5), requiring an expansion to (batch, 28, 120, 120). Since direct convolution is insufficient to upscale such small spatial dimensions, the model first applies a 3×3 convolution with padding=1, preserving the 28 feature channels while extracting relevant patterns. The key step involves bilinear upsampling, which smoothly scales the data from (5×5) to (120×120).


  - **Soil Composition Data**: SoilCompositionNet() handles high-dimensional soil composition data, which starts with an input shape of (batch, 18, 1306, 464) and is downsampled to (batch, 18, 120, 120). The model applies a 7×7 convolution with stride=4, increasing feature maps from 18 to 32 while significantly reducing spatial dimensions. A subsequent 3×3 convolution with stride=2 restores the feature representation to 18 channels, ensuring that soil composition types remain distinguishable. Finally, adaptive average pooling resizes the output to (120,120).


*Output matrix stays the same as used in Approach 1. These steps can be found in the Final_Modeling/MutliModalFCNew.py*

We tried 2 versions of the main predition model, one with Fully Convolutional Networks (FCNs) and another with U-Net. U-Net model outperformed FCN model so we chose to work with U-Net. The U-Net architechture used is defined in the file *Final_Modeling/Modeling_Final_Version_Feb22.ipynb*

For the loss function, we used Weighted Cross Entropy Loss where weights are the inverse of each class' frequency. 
## Evaluation Metrics

Following metrics are used to evaluate the performance of our approaches:

- **Pixel Accuracy Calculation**: Computes the proportion of correctly classified pixels by comparing predictions with ground truth. Uses (pred_flat == true_flat).float().mean().item() to calculate accuracy.

- **Confusion Matrix**: Generates a confusion matrix to show how predictions align with true labels. Uses confusion_matrix() with class labels [0, 1, 2] to evaluate per-class performance.

- **Precision, Recall, and F1 Score (Per Class)**: Computes precision, recall and F1 score for each class using precision_score(), recall_score(), and f1_score() with average=None for per-class results

- **Intersection over Union (IoU) for Each Class**: Computes intersection and union for each class. Uses (intersection / union).item() formula. Handles division by zero by appending NaN if a class is missing in true labels.
- **Dice Coefficient for Each Class**: Uses (2 * intersection) / (sum of predicted + true pixels + epsilon). Adds a small eps=1e-6 to prevent division by zero. This metric is similar to the F1 score.

- **Balanced Accuracy Score**:Computes Balanced Accuracy, which adjusts for class imbalance. Uses balanced_accuracy_score() from sklearn.metrics.
## Best Results

We got the best results using the U-Net model with data preprocessing approach 2. Here are the scores for each of the above evaluation metrics:

- Pixel Accuracy: 0.71
- Precision per class: [0.99, 0.07, 0.11]
- Recall per class: [0.71, 0.66, 0.80]
- F1 Score per class: [0.83, 0.13, 0.20]
- IoU per class: [0.71, 0.06, 0.11]
- Dice Coefficient per class: [0.83, 0.13, 0.20]
- Balanced Accuracy: 0.73

The model demonstrates strong overall performance, achieving a pixel accuracy of 71.4%, which indicates that the majority of pixels are correctly classified. 

Despite the challenges posed by high class imbalance, the model shows promising recall values, particularly for classes 1 and 2, with recall scores of 66.9% and 80.9%, respectively. This suggests that the model effectively identifies instances of these less frequent classes, even if precision remains an area for improvement.

The Intersection over Union (IoU) and Dice Coefficient scores reinforce this trend, with the dominant class achieving 71.0% IoU and 83.1% Dice score, indicating strong segmentation performance. For the minority classes, while the scores are lower, they serve as a valuable baseline for future optimization, particularly through data balancing strategies or loss function adjustments.

Moreover, the balanced accuracy of 73.0% suggests that the model does not entirely favor the majority class, meaning it retains some ability to differentiate between all categories. 

Overall, these results provide a solid starting point for segmentation improvements, with strengths in majority-class prediction and recall for minority classes. By incorporating class rebalancing techniques, loss weighting, or data augmentation, the model’s performance across all categories can be further enhanced, leading to more precise and reliable segmentation outcomes.

## Potential System Deployment in Future


Our landslide detection model holds significant potential for real-world deployment in two key scenarios:
 - First, an on-demand system could be developed where users input specific latitude and longitude coordinates, prompting the model to retrieve relevant geospatial and environmental data to generate a detailed landslide susceptibility map for that location. This would be particularly useful for researchers, urban planners, and developers assessing risk before initiating projects.
- Second, a fully automated risk management system could be implemented, where the model continuously collects real-time data from various sources to analyze landslide probabilities. In cases where the system detects a high likelihood of a landslide, it could issue early warnings to relevant authorities and communities, enabling proactive mitigation measures. By integrating this model into disaster preparedness frameworks, we can enhance risk assessment capabilities and improve response strategies, ultimately reducing the impact of landslides on vulnerable regions.  
