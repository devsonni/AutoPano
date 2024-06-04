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
```

You can test the given data set using following command or can create custom dataset as follows:    

To run stitching:    

```bash
python3 Wrapper.py --Path $your-path-to-dataset --Scale $scale-factor
```

Here, 
- Path parser will take the path from the current directory to dataset location    
- Scale parser can be used if your image size if big, and you want to reduce the size

To create custom dataset:    
- create a dataset folder and add all images, name all images in numberical order (be carful of order)    

Example of running wrapper on custom dataset:    
```bash
python3 Wrapper.py --Path ../Data/Set1 --Scale 4
```

