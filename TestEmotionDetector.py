import cv2
import numpy as np
from keras.models import model_from_json
import time

start_time = time.time()
emotion_count = dict()

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json file and create model

json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

#load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded Model From disk")

# start the web cam
#cap = cv2.VideoCapture(0)

#file path =  "C:\\Users\\LENOVO\\Desktop\\Test_video\\sample_2.mp4"
file_path = input("Enter Your file path: ")

cap = cv2.VideoCapture(file_path)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Detect faces avaliable on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take all the faces available on the camera and process it
    for(x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        #print(emotion_dict[maxindex])

        #Write function to write the emotion in txt file
        def write():
            with open("test_Result.txt", 'a') as f:
                f.write(max_emotion_count)
                f.write("\n")

        # Time depedent code to claculate the max emotion in 1 sec
        if ((time.time() - start_time) <= 1):
            emotion_count[emotion_dict[maxindex]] = emotion_count.get(emotion_dict[maxindex],0) + 1
        else:
            print(emotion_count)
            if(len(emotion_count) > 0):
                max_emotion_count = max(zip(emotion_count.values(), emotion_count.keys()))[1]
                print("****************", max_emotion_count , "********************")
                write()
                emotion_count.clear()
            start_time = time.time()

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
