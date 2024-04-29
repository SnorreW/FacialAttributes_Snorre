This is a program that uses Facer and DeepFace for Facial Attribute Classification.

1. Feature_embeddings_lfw - contains JSON files of the feature embeddings of each image in the LFW Dataset. For faster computing. Used to generate statistics.
2. Feature_embeddings_sm2 - contains JSON files of the feature embeddings of each image in the selfmadefolder2. For faster computing. Used to test performance.
3. lfw-deepfunneled - the lfw image set.
4. selfmadefolder2 - contains images gathered from the internet. Make sure the images are in ascending order from 1 to 48 and not 1,10,11,12...
5. failure_to_detect_lfw - json file of the number of times Facer and DeepFace failed to identify a face.
6. main.py - used to generate feature embeddings to JSON files given a file path to a folder with images.
7. performance2.py - generates heatmaps of the results in feature_embeddings_sm2.
8. references.txt - contains links to all the images in selfmadefolder2.
9. requirements.txt - necessary dependencies to run the scripts.
10. statistics_dataset.py - generates statistics of the feature embeddings in feature_embeddings_lfw.
11. truth_labels_sm2.csv - ground truth labels of the images in selfmadefolder2.
