This is a fork from https://github.com/jonasbtn/helipad_detection to use the dataset for yolo training. This fork generates yolo compatible dataset and labels. 
Follow the steps to generate dataset: 
1. To download images from google map we need google map static api key.

Link on how to get an API key: https://developers.google.com/maps/documentation/maps-static/get-api-key

popilate api.csv with the api key. 

2. To generate Ensemple training dataset run csv_to_meta.py 
