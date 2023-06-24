# FaceVerification


Firstly in order to run this project you have to install the libraries in the requirements.txt files provided in the directory. Given that the requirements are met you have to follow the steps below:

1. Download the data from [University of Massachusetts](http://vis-www.cs.umass.edu/lfw), where from the many options we used the [GzippedTar File](http://vis-www.cs.umass.edu/lfw/lfw.tgz).
2. Add the data folder in the same directory.
3. Run the set_up.py notebook that is going to set up the data structure automatically.
4. During the execution of the script it will use OpenCv's VideoCapture model to open the camera of your machine. (Default device is 0, which may vary in based on the machine)
5. The first  time you have to press 'a' in order to save the frame in the anchor folder. When ready press 'q' to close the window and move on to the next task.
6. The second time you have to press 'p' in order to save the frame in the positive folder. When ready press 'q' to close the window and conclude the set-up of the project.
7. **SOS** for the steps 5-6 it is adviced to check the number of frames you have saved in each folder. A good number is 300-500 images with the aim of creating enough pictures with the data augmentation step right after.
8. Execute the train.ipynb file to train the model and get the products of the training like checkpoints, model.h5 and accuracy results in a txt format.
9. Execute the demo_set_up.ipynb file to create the app folder that is going to create the application environment.
10. Lastly run the verification.py file in order to test the model.
