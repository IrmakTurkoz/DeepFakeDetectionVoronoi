# DeepFakeDetectionVoronoi
Face Anomally Detection with Voronoi Regions as Features
To execute this program (this may take hours.): 

1) Download the face forensics++ dataset with faceforensics_download_v4. 
You can find the instructors from https://github.com/ondyari/FaceForensics

2)Download the shape predictor from dlib. 
You can find the instructors from http://dlib.net/

3) To extract the features with voronoi regions use the voronoi regions main, uncomment the 
#     fileDirectory = "New_new_Sample_set/val/manipulated_sequences/features/"

#      destination = "New_new_Sample_set/val/manipulated_sequences/padded_features/"
#      videos = [f for f in os.listdir(fileDirectory) if ("mp4") in f]
#      print(videos)
#      model.extract_features(fileDirectory,destination,videos,True,show_results =False,frame_rate = 50)
#      model.pad_images(fileDirectory,destination)


Use your directory of where you have saved your files. You should also give the shape predictor from step 2 
as an argument to the main.

4) After extraction you can use the model. ( If you have keras. )
If you don't have Keras tensorflow, and virtual envoirenment, follow the 
instructions on https://www.tensorflow.org/install

5) You can use either one of the mains in the voronoi region or facial regions.

6) There is also a model2.ipynb in facial regions, where you can find the already trained models outputs.

