# AutoPano - Panorama Stitching    
The panorama stitching project uses SIFT feature descriptors for key point matching and RANSAC for outlier rejection. The images are blended with distance transformation, resulting in a seamless and coherent panoramic image.     

### Requirements

To run the panorama stitching project, ensure you have the following libraries installed:

- numpy
- opencv-python (cv2)
- argparse
- os (standard library)
- math (standard library)
- glob (standard library)

You can install the required libraries using pip:

```bash
pip install numpy opencv-python argparse

