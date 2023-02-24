# coding: utf-8

from utils_st_presentation import new_exercise, transform_keypoints
import glob
import cv2
import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import time
import tempfile
from sklearn.metrics.pairwise import cosine_similarity


import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
########################## 0.
### data preparation - define useful data


########################## AA
st.title('Form of execution')

############## A. SELECTING EXERCISES FOR PRACTICE
#################################################
#################################################

### get list of excercises from path
uploaded_exercise = [x.split('\\')[-1] for x in glob.glob('best_forms_collection' + '/*')]
    
# st.write(tuple(uploaded_exercise))

exc_option = st.selectbox('Select the excercise you want to perform!', ('Select an exercise',) + tuple(uploaded_exercise), key=4)

if exc_option != 'Select an exercise':

    # path = r'C:/Users/manu_/FP/Presentation/best_forms_collection/' + exc_option
    
    
    ############ instructional video from database                                ############### drive link required
    ################################
    ################################
          
    path = 'best_forms_collection' + '/'+ exc_option
    
    full_path = glob.glob(path + '\*')
    
    st.write('Please record yourself doing the exercise like the person in the video')
    # st.write(full_path[0])
    
    width = st.slider(
    label="Width", min_value=10, max_value=100, value=50, format="%d%%"
)

    side = max((100 - width) / 2, 0.01)
    _, container, _ = st.columns([side, width, side])
    container.video(data=full_path[0])



    
    

    # st.video(full_path[0]) # index is coordinated with the upload structure for coaches' files

    # testing the saved frames
    with open(full_path[1], 'rb') as f:
        coach_frames = pickle.load(f)
    
    with open(full_path[2], 'rb') as f:
        coach_keypoints = pickle.load(f)
            
    
    # access keypoints of best_form and save as fattened 2d list for cosine similarity
    ##################################
    
    with open(full_path[2], 'rb') as f:
        ref_keypoints = pickle.load(f)
    ref_frame_KP = transform_keypoints(ref_keypoints[0])
    

    ############## D.2  USER UPLOAD VIDEO
    #####################################
    #####################################
    user_upl = None
    
    while user_upl == None:
        if st.radio('Share your attempt', ['Upload video','Record attempt'], key=5) == 'Upload video':
            user_upl = st.file_uploader("Upload your video", ['mp4', 'avi'])


            
            if user_upl:
                # lists for saving keypoints and frames
                vid_keypoint_list = []
                frames_usr_vid = []      # not necessary


                ############ change of dtype/reference for video call
                #####################################################
                #####################################################

                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(user_upl.read())
                video_path_user = tfile.name

                ############## intialize video_capture
                #####################################################
                #####################################################

                vid_cap = cv2.VideoCapture(video_path_user)
                # st.write(f'vid_cap shape {}')
#                 shape = (vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH), vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT), user_upl)
                
                pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
                st_frame = st.empty()

    #             fr_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    
                count = 0
                idx = 0
                ret = True
                user_v_i = True
                while ret:
                    ret, frame = vid_cap.read()
                    if (type(frame) != type(None)):     # there can be frames with value None

                        # recolor image to be in RGB format for the pose estimation
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False # to save memory reduce attributes

                        # Make detection and save
                        ##########################
                        results = pose.process(image)
                        vid_keypoint_list.append(results.pose_landmarks)

                        # Test similar position with cosine similarity
                        if idx == 0:
                            try:
                                user_fr_KP = transform_keypoints(results.pose_landmarks) # makes 2d array
                                sim = cosine_similarity(user_fr_KP, ref_frame_KP)
                                if (sim > 0.9): # finding the first similar position above threshhold
                                    idx = count
                            except:
                                pass
                            count += 1

                        # recolor image to RGB
                        image.flags.writeable = True
                        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        frames_usr_vid.append(image.copy())

                        # Render detections to show results of pose.process

                        ##############
                        ##############
                        ##############
                        ##############
                        ##############
                        ##############
                        ##############
                        ##############
                        ##############
                        ##############

                        # coach_keypoints
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                 mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                        image = cv2.resize(image, (600, 600))

                        # st_frame.image(image)

                        #### THRESHHOLD
                        ###############
                        ###############
                    if cv2.waitKey(1) and (vid_cap.get(cv2.CAP_PROP_POS_FRAMES)>90) and (vid_keypoint_list[-1] == None):    
                        user_upl = None
                        vid_cap.release()
                        # cv2.destroyAllWindows()
                        user_v_i = False
                        # message to user --> break the whole process of user feedback                         #####
                        st.write('There is no one in your video. Make sure to start recording with a person in the frame')
                        del vid_keypoint_list, frames_usr_vid
                        break


                        # if cv2.waitKey(1) and (vid_cap.get(cv2.CAP_PROP_POS_FRAMES) == 150):    ### change here
                        #     vid_cap.release()
                        #     # cv2.destroyAllWindows()
                        #     user_v_i = False
                        #     st.write('The person in the video has to match the position sooner. Too late!')
                        #     del vid_keypoint_list, frames_usr_vid
                        #     break
                    if ret != True:
                        vid_cap.release()
                        break

                    if cv2.waitKey(1) and (count == cv2.CAP_PROP_FRAME_COUNT):
                        st.write('The person in the video has to match the position sooner. Too late!')
                        user_upl = None
                        break


        # else: # if st.radio('Share your attempt', ['Upload video','Record attempt']) == 'Record attempt':
        #     if st.button("Record a short video for five seconds"):
        #         st.write('Not yet available, if you like the idea, support with coding hours or a donation ;)!')

    #             # save keypoints and frames
    #             vid_keypoint_list = []
    #             frames_usr_vid = []

    #             # intialize video_capture(0)
    #             vid_cap = cv2.VideoCapture(video_path_user)
    #             pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    #                 # '.Pose' method accesses a pose estimation model,
    #                 # min_detection_confidence parameter for accuracy 
    #                 # min_tracking_confidence 

    #             st_frame = st.empty()

    #             fr_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    #             ret = True
    #             while ret:
    #                 ret, frame = vid_cap.read()

    #                 # recolor image to be in RGB format for the pose estimation
    #                 image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                 image.flags.writeable = False # to save memory reduce attributes

    #                 # Make detection
    #                 results = pose.process(image)
    #                 vid_keypoint_list.append(results.pose_landmarks)
    #                 # recolor image to RGB
    #                 image.flags.writeable = True
    #                 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #                 frames_usr_vid.append(image)

    #                 # Render detections to show results of pose.process
    #                 mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    #                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
    #                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    #                 image = cv2.resize(image, (600, 600))

    #                 st_frame.image(image)

    #                 #### THRESHHOLD
    #                 if cv2.waitKey(1) and (vid_cap.get(cv2.CAP_PROP_POS_FRAMES)>90) and (vid_keypoint_list[-1] == None):    

    #                     vid_cap.release()
    #                     cv2.destroyAllWindows()

    #                     # message to user --> break the whole process of user feedback                         #####
    #                     print('There is no one in your video. Make sure to start recording with a person in the frame')
    #                     del vid_keypoint_list, frames_usr_vid
    #                     break


    #                 if cv2.waitKey(1) and (vid_cap.get(cv2.CAP_PROP_POS_FRAMES) == fr_count):
    #                     break


    #             vid_cap.release()
    #             cv2.destroyAllWindows()



    #             #########  running videos user video to display
    #             ###########################################
    #             ###########################################

    #             frames_usr_vid = []
    #             frames_usr_vid2 = []

    #             ### displaying relevant timeframes
    #             vid_cap = cv2.VideoCapture(video_path_user)
    #             vid_cap2 = cv2.VideoCapture(full_path[0])
    #             pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


    #             vid_cap.set(cv2.CAP_PROP_POS_FRAMES, idx) #  taking index from step 5
    #             st_time = time.time()
    #             st_frame = st.empty()

    #             fr_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    #             ret = True
    #             while ret:
    #                 ret, image = vid_cap.read()
    #                 ret2, frame2 = vid_cap2.read()
    #                 act_frame = cv2.CAP_PROP_POS_FRAMES
    #                 # recolor image to be in RGB format for the pose estimation
    # #                 image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # #                 image.flags.writeable = False # to save memory reduce attributes

    # #                 # Make detection
    # #                 results = pose.process(image)
    # #                 vid_keypoint_list.append(results.pose_landmarks)

    # #                 # recolor image to RGB
    # #                 image.flags.writeable = True
    # #                 # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #                 # # RENDER OWN LANDMARKS OR
    #                 # mp_drawing.draw_landmarks(image, vid_keypoint_list[act_frame], mp_pose.POSE_CONNECTIONS,
    #                 #                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
    #                 #                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    #                 # RENDER REF LANDMARKS
    #                 try: 
    #                     mp_drawing.draw_landmarks(frame, ref_keypoints[act_frame], mp_pose.POSE_CONNECTIONS,
    #                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
    #                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    #                 except:
    #                     pass

    #                 image = cv2.resize(image, (224, 224))
    #                 frame2 = cv2.resize(frame2, (224, 224))
    #                 # st_frame.image(image)
    #                 frames_usr_vid.append(image)
    #                 frames_usr_vid2.append(frame2)
    #                 #### THRESHHOLD
    #                 now_time = time.time()
    #                 if (now_time - st_time)>4:
    #                     vid_cap.release()
    #                     vid_cap2.release()
    #                     cv2.destroyAllWindows

    #                     st.write('We just finished comparing')
    #                     break
    #             imgs_display1 = np.stack(frames_usr_vid)
    #             imgs_display2 = np.stack(frames_usr_vid2)
        break
        
    if user_upl != None:
        st.write(len(frames_usr_vid))

        col1, col2, col3 = st.columns(3) # from now on, we can add after a dot  functions of streamlit to col1 and col2 
                    # instead of st.write() we can also use col1.write()


        col1.header("Here is your performance and it's skeleton" )

        interv_ = 5
        int_split = int(len(coach_keypoints)/interv_)
        for i in range(0, len(coach_keypoints),  int_split):
            if i < len(frames_usr_vid):
                # st.write(f' this is the coach_keypoints type: {type(coach_keypoints[i-idx])}')
                display = frames_usr_vid[i]
                display = cv2.resize(display, (300,300))
                ###

                mp_drawing.draw_landmarks(display, vid_keypoint_list[i], mp_pose.POSE_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=2),
                                         mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=2))


                col1.image(display)




        col2.header('Here is your performance with overlay')
        for i in range(idx,idx + len(coach_keypoints),  int_split):
            if i < len(frames_usr_vid):
                display2 = frames_usr_vid[i]
                display2 = cv2.resize(display2, (300,300))
                mp_drawing.draw_landmarks(display2, coach_keypoints[i], mp_pose.POSE_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=2),
                                         mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=2))
                

                col2.image(display2)
        col3.header('Example of good form for exercise')
        for i in range(0,len(coach_keypoints), int_split):
            if i < len(coach_frames):
                display3 = cv2.resize(coach_frames[i], (300,300))
                col3.image(display3)

    if user_upl != None:
        st.write(f'the total number of frames of user video {len(frames_usr_vid)}')
        sld = st.slider('Select frame', min_value = 0, max_value = len(coach_keypoints) - 1, step = 1)
        st.write(idx) #wrong idx 
        slide_display = frames_usr_vid[sld] # + idx]

        if st.radio('Select the skeleton', ['Your form','Form of example']) == 'Your form':
            mp_drawing.draw_landmarks(slide_display, vid_keypoint_list[sld+idx], mp_pose.POSE_CONNECTIONS,
                     mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=3, circle_radius=5),
                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=5, circle_radius=2))

        else:
            mp_drawing.draw_landmarks(slide_display, coach_keypoints[sld], mp_pose.POSE_CONNECTIONS,
                     mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=3, circle_radius=5),
                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=5, circle_radius=2))

        slide_display = cv2.resize(slide_display, (600,600))
        st.image(slide_display)
                
                

        
    ############## D.3. USER RECORDS VIDEO
    
    
            
            
            #######################   display results


            # col1, col2 = st.columns(2) # from now on, we can add after a dot  functions of streamlit to col1 and col2 
            #                         # instead of st.write() we can also use col1.write()

            # col1.header('Here is your performance')
            # col1.image(image)
            # col2.header('Example of good form for excecize')
            # col2.image(image_pose)

            # col2.image(image)

            # A. model prediction for the frames of the video
            # A.1 predicting values for the frame of the video
            # for frame in image[]'
                # model.predictframe
                    # or maybe directly saving all frame predictions onto a vide
                # 

            # A.2 choosing the predicted output for the chosen excercise under
            
# keep working with the the the following filenames
        # for frames:     frames_usr_vid
        # for keypoints:  

        
# import utils_st_presentation

path = None






#############################################################
########### CHANGES - saving best_form with keypoints


### EE. Coaches upload !! finish adding descriptions

credentials = st.text_input('Are you a coach and want to upload an exercise with best execution? Give in your credentials')
# x ='yeah :)'
if credentials == "1234":
    name = st.text_input('What is the name of the exercise or skill do you want to upload?')
    if name:
        # if st.button('Upload your video', key=6):
        upl_vid = st.file_uploader("Upload the best video you have. consider the perspective", ['avi','mp4'])
        if upl_vid:
            st.write('thanks for contributing')
            st.write(type(upl_vid))
            st.write(upl_vid)
            # st.write(upl_vid.getvalue()) # pc always freezes with this
            # read file as bytes:
            st.write(f'this is the name: {name}')
            st.write(upl_vid.name)
            st.write(f'this is the name: {name}')
            # st.write(upl_vid.read())
            
            ############ change of dtype/reference for video call
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(upl_vid.read())
            video_path_coach = tfile.name
            
            #############   calling the video
            
            # prepare directories to save files
            upload = new_exercise(name, video_path_coach)
            upload.create_path()
            
            # save keypoints and frames
            vid_keypoint_list = []
            frames_coach_vid = []
            
            # initializing the capture

            vf = cv2.VideoCapture(video_path_coach)
            
            # CHANGE ## videoWriter
            frame_width = 600    # int(vf.get(3))
            frame_height = 600     # int(vf.get(4))
            size = (frame_width, frame_height)
            # # https://drive.google.com/drive/folders/1gezT8IIQrOn43tprINQAX4vNfvwJ811i?usp=sharing
            # vr = cv2.VideoWriter(f'best_forms_collection/{name}/{name}.mp4',
            #                      cv2.VideoWriter_fourcc(*'MJPG'),
            #                      10, size)
            # ### - END CHANGE ##
            
            pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            
            stframe = st.empty()

            while vf.isOpened():
                ret, frame = vf.read()
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                
                im2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(im2)
                vid_keypoint_list.append(results.pose_landmarks)        # mind: user code requires vid_keypoint_list[0].landmark to access landmarks
                
                im2 = cv2.resize(im2, (600, 600))
                frames_coach_vid.append(im2.copy())
                # Render detections to show results of pose.process
                mp_drawing.draw_landmarks(im2, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                         mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                stframe.image(im2)
                # st.write(f'datatype of landmarks displayed: {type(results.pose_landmarks)}')
                # im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)
                
                # CHANGE ##
                # vr.write(im2)
                ## - END CHANGE ##

            vf.release()
            # CHANGE ##
            # vr.release()
            ## - END CHANGE ##
            cv2.destroyAllWindows()

            ############ saving of video file   
            st.write(f'datatype of my landmarks list:{vid_keypoint_list} and element vid_keypoint_list[]: {vid_keypoint_list}')
            upload.load_file()
            upload.pickle_save(vid_keypoint_list, 'KeyPoints')
            upload.pickle_save(frames_coach_vid, 'frames')
            
            st.write('Upload Successful')
                
