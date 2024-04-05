### Using feature extraction module

**Example 1** 

Extracting certain Pixel intensity features from images 
- Load the class
  - `from preprocessing.feature_extraction import ImageFeatureExtractor`
- Load the image with opencv or pillow
  - `from PIL import Image`
  - `img = Image.open('/parent/./dir/10253_idx5_x1001_y1001_class0.png')`
- Create feature extraction object
  - `feature_extraction = ImageFeatureExtractor(features={'MPI': False})`
  - Note: defaults to creating all features, just setting MPI as false as an example
- Call the class with the image
  - `features = feature_extraction(img)`


**Example 2** 

Extracting details from image path for this [dataset](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images/data) 
- Load the function
  - `from preprocessing.feature_extraction import extract_details_from_path`
- pass the image file path into the function
  - `img_path = '/parent/./dir/10253_idx5_x1001_y1001_class0.png'`
  - `extract_details_from_path(img_path)`