# visual_feedback_for_excercise
Project for  uploading/recording your own execise execution and getting a comparison of form with a example of best practice execution.

The goal is an application for comparing movements executions of a movement or technique in a given sport or discipline (e.g. gym exercises or Parkour).
The user experience is designed as follows:

1) Users wanting to get feedback about a move upload a video of themselves performing these movement. 
2) The model takes the given input video and chooses another video of the same movement with good execution. There is a constraint on the videos of good execution: These videos contents are reduced to only the execution of the movement from start to end! (*) 
3) There are two outputs of the app: 
    A) On the one hand there are two columns of images. One column displays the user execution. For different phases of the movement there is a picture with the bodyposture drawn on it. The other column contains respective frames from the video with good execution, for comparison. 
    B) On the other hand there is a slider that allows you to slide through all the frames of the user video containing only the movement. Additionally to sliding, the user can overlay choose a posture skeleton to overlay on the images: either the posture skeleton of himself, or the posture skeleton of the "good example video" so he can compare his own posture with the psoture of good example for every frame of the movement. 
    
As of now there is only a minimum viable product of this app running I made for my final project. Till now it only run on local host with streamlit. 

Tools: 
  - MediaPipe: For pose estimation
  - OpenCV: For handling videos
