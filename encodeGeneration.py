import cv2
import os
import face_recognition
import pickle

# Importing students images
folderPath = 'Images'
PathList = os.listdir(folderPath)
# print(PathList)
imageList = []
ids=[]
for path in PathList:
    imageList.append(cv2.imread(os.path.join(folderPath, path)))
    ids.append(os.path.splitext(path)[0])
# print(imageList)
# print(ids)

def findEncodings(imageList):
    enc=[]
    for img in imageList:
        # 1. convert to RGB
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 2. find the encodings
        encode=face_recognition.face_encodings(img)[0]
        enc.append(encode)

    return enc
print("Begin")
encodeListKnown=findEncodings(imageList)
# print(encodeListKnown)
encodeWithIds=[encodeListKnown,ids]

print("Done")

file=open("EncodeFile.p",'wb')
pickle.dump(encodeWithIds,file)  # dump this in file
file.close()
print("File Saved")

