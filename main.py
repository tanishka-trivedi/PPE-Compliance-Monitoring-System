import pickle
import face_recognition
import cv2
import os

cap=cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)

imgBackground=cv2.imread('Resources/background.png')

# Importing mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imageModeList = []
for path in modePathList:
    imageModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# print(len(imageModeList))
# import/load the encoding file
file= open('EncodeFile.p','rb')
encodeWithIds = pickle.load(file)
file.close()
encodeListKnown,ids = encodeWithIds
# print(ids)
while True:
    success, img=cap.read()

    imgS=cv2.resize(img,(0,0),None,0.25, 0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    faceCurFrame=face_recognition.face_locations(imgS)
    encodeCurFrame=face_recognition.face_encodings(imgS,faceCurFrame)


    imgBackground[162:162+480,55:55+640]=img
    imgBackground[44:44+633, 808:808 + 414] = imageModeList[3]

    for encodeFace, facLoc in zip(encodeCurFrame, faceCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace) # lower the distance- better
        print("Matches: ",matches)
        print("Distance: ",faceDis)
    # cv2.imshow("Webcam",img)
    cv2.imshow("Face Attendance",imgBackground)
    cv2.waitKey(1)
