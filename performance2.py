import matplotlib.pyplot as plt
import json
import os
import time
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#This script tests the performance of the program on the image set that was manually labeled.
#Uses confusion matrices to visualize.
path = "feature_embeddings_sm2"

#gender
facer_gender_pred = []
df_gender_pred = []
combined_gender_pred = []
#ethnicity
ethn_pred = []
#glasses
glasses_pred = []
#happy
facer_happy_pred = []
df_happy_pred = []
combined_happy_pred = []
#age
age_pred = []

# truth labels
truth_labels_df = pd.read_csv("truth_labels_sm2.csv", delimiter=";")
print(truth_labels_df["name"][0])
current = 0
start = time.perf_counter()

for i in range(len(os.listdir(path))):
    i+=1

    file = open(path + "/" + str(i) + ".jpg.json")
    print(path + "/" + str(i) + ".jpg.json")
    data = json.load(file)



    #Predicted gender df, facer and combined
    df_gender_pred.append(data["df_gender"][0].lower())
    if df_gender_pred[current] == "w":
        df_gender_pred[current] = "f"

    if data["Male"] >= 0.5:
        facer_gender_pred.append("m")
    if data["Male"] < 0.5:
        facer_gender_pred.append("f")

    combination_gender = 0
    if data["df_gender"] == "Man":
        combination_gender = (data["Male"] + (data["df_gender_prob"] / 100)) / 2
    if data["df_gender"] == "Woman":
        combination_gender = (data["Male"] + (1 - (data["df_gender_prob"] / 100))) / 2

    if combination_gender >= 0.5:
        combined_gender_pred.append("m")

    if combination_gender < 0.5:
        combined_gender_pred.append("f")

    #Predicted ethnicity
    ethn_pred.append(data["df_race"])


    #Predicted glasses in an image
    if data["Eyeglasses"] >= 0.5:
        glasses_pred.append(True)
    else:
        glasses_pred.append(False)

    #Predicted happy people
    if data["Smiling"] >= 0.5:
        facer_happy_pred.append(True)
    else:
        facer_happy_pred.append(False)
    if data["df_emotion"] == "happy":
        df_happy_pred.append(True)
    else:
        df_happy_pred.append(False)

    # combination happy
    combination_happy = 0
    if data["df_emotion"] == "happy":
        combination_happy = (data["Smiling"] + (data["df_emotion_prob"] / 100)) / 2
    if combination_happy >= 0.5:
        combined_happy_pred.append(True)
    else:
        combined_happy_pred.append(False)

    #Predicted young or old
    if data["Young"] >= 0.5:
        age_pred.append("young")
    else:
        age_pred.append("old")

    current += 1
    file.close()



stop = time.perf_counter()
final_1 = stop - start

#Confusion matrix for genders
gender_truth =truth_labels_df["gender"].to_numpy()
facer_gender_matrix = confusion_matrix(gender_truth, facer_gender_pred)
facer_gender_display = ConfusionMatrixDisplay(confusion_matrix=facer_gender_matrix, display_labels=["Females","Males"])
facer_gender_display.plot()
plt.title("Facer gender prediction")
plt.show()
df_gender_matrix = confusion_matrix(gender_truth, df_gender_pred)
df_gender_display =ConfusionMatrixDisplay(confusion_matrix=df_gender_matrix, display_labels=["Females","Males"])
df_gender_display.plot()
plt.title("Deepface gender prediction")
plt.show()
combined_gender_matrix = confusion_matrix(gender_truth, combined_gender_pred)
combined_gender_display = ConfusionMatrixDisplay(confusion_matrix=combined_gender_matrix, display_labels=["Females","Males"])
combined_gender_display.plot()
plt.title("Facer/DF gender prediction")
plt.show()

#Confusion matrix for ethnicities
ethn_truth = truth_labels_df["race"].to_numpy()
ethn_matrix = confusion_matrix(ethn_truth, ethn_pred)
display = ConfusionMatrixDisplay(confusion_matrix=ethn_matrix, display_labels=["Asian","Black","Indian","Latino","Mid.Eastrn","White"])
display.plot()
plt.title("Ethnicity prediction")
plt.show()

#Confusion matrix glasses
glasses_truth = truth_labels_df["glasses"].to_numpy()
glasses_matrix = confusion_matrix(glasses_truth, glasses_pred)
glasses_matrix_display = ConfusionMatrixDisplay(confusion_matrix=glasses_matrix, display_labels=["No glasses", "Glasses"])
glasses_matrix_display.plot()
plt.title("Glasses prediction")
plt.show()

#Confusion matrix happiness
happy_truth = truth_labels_df["happy"].to_numpy()
facer_happy_matrix = confusion_matrix(happy_truth, facer_happy_pred)
facer_happy_matrix_display = ConfusionMatrixDisplay(confusion_matrix=facer_happy_matrix, display_labels=["Not happy", "Happy"])
facer_happy_matrix_display.plot()
plt.title("Facer happiness prediction")
plt.show()
df_happy_matrix = confusion_matrix(happy_truth, df_happy_pred)
df_happy_matrix_display = ConfusionMatrixDisplay(confusion_matrix=df_happy_matrix, display_labels=["Not happy", "Happy"])
df_happy_matrix_display.plot()
plt.title("DeepFace happiness prediction")
plt.show()
combined_happy_matrix = confusion_matrix(happy_truth, combined_happy_pred)
combined_happy_matrix_display = ConfusionMatrixDisplay(confusion_matrix=combined_happy_matrix, display_labels=["Not happy", "Happy"])
combined_happy_matrix_display.plot()
plt.title("Facer/DF happiness prediction")
plt.show()

#Confusion matrix for Old and young people
age_truth = truth_labels_df["age"].to_numpy()
age_pred_matrix = confusion_matrix(age_truth, age_pred)
age_pred_matrix_display = ConfusionMatrixDisplay(confusion_matrix=age_pred_matrix, display_labels=["Old", "Young"])
age_pred_matrix_display.plot()
plt.title("Old and young prediction")
plt.show()

#Gender results
print("Computational time: " + str(final_1))

