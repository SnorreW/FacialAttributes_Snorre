import facer
import torch
import os
import time
from deepface import DeepFace
import json

print("pytorch version", torch.__version__)
print("is cuda enabled? ", torch.cuda.is_available())
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

#Races
asian = 0
indian = 0
black = 0
white = 0
middleeastern = 0
latino = 0
##errors
facer_error = 0
deepface_error = 0

#Define the path of the folder that you want to analyze
path_person = "selfmadefolder2/"
start = time.perf_counter()
face_detector = facer.face_detector('retinaface/mobilenet', device=device)
face_attr = facer.face_attr("farl/celeba/224", device=device)
tot = 0
# https://github.com/FacePerceiver/facer
# There is a loop in a loop because of how the LFW dataset is structured. The LFW folder contains folders with images. A new dataset needs to be structured this way aswell.
for i in os.listdir(path_person):
    for j in os.listdir(path_person + i):
        img_object = {"name": j}
        try:

            image = facer.hwc2bchw(facer.read_hwc(path_person + i + "/" + j).to(device=device))
            # face detection
            faces = face_detector(image)
            faces = face_attr(image, faces)
            labels = face_attr.labels
            face1_attrs = faces["attrs"][0]
            for prob, label in zip(face1_attrs, labels):
                img_object[label] = prob.item()

            race = DeepFace.analyze(img_path=path_person + i + "/" + j, actions=["race", "gender", "age", "emotion"])
            #print(race[0])
            img_object["df_race"] = race[0]["dominant_race"]
            img_object["df_race_prob"] = race[0]["race"][race[0]["dominant_race"]]
            img_object["df_gender"] = race[0]["dominant_gender"]
            img_object["df_gender_prob"] = race[0]["gender"][race[0]["dominant_gender"]]

            img_object["df_male_prob"] = race[0]["gender"]["Man"]
            img_object["df_female_prob"] = race[0]["gender"]["Woman"]

            img_object["df_age"] = race[0]["age"]
            img_object["df_emotion"] = race[0]["dominant_emotion"]
            img_object["df_emotion_prob"] = race[0]["emotion"][race[0]["dominant_emotion"]]

            img_object["df_happy"] = race[0]["emotion"]["happy"]

            tot += 1
            if tot == 100:
                stop = time.perf_counter()
                t = stop - start
                print("average per image: ", str(t/100))
                print("Whole dataset: ", str((t/100) * 13000))
            print(img_object)
            #Define where to store the feature embeddings.
            with open("feature_embeddings_sm3/" + j + ".json", "w") as file:
                json.dump(img_object, file)
        except KeyError:
            print(path_person + i + "/" + j)
            print("No face detected, Facer")
            facer_error += 1
        except ValueError:
            print("no face detected, DeepFace")
            print(j)
            deepface_error += 1

stop = time.perf_counter()
time = stop - start

failure_to_detect = {"facer": facer_error, "deepface": deepface_error, "total_images": tot}
with open("failure_to_detect.json", "w") as errors:
    json.dump(failure_to_detect, errors)

print(str(time))



