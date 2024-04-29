import matplotlib.pyplot as plt
import json
import os
import time


#This script uses the feature embiddings that was created for the LFW dataset and procduces some statistics about the dataset.
path = "feature_embeddings_lfw"

#gender
male = 0
female = 0
#ethnicity
asian = 0
indian = 0
black = 0
white = 0
middleeastern = 0
latino = 0
#glasses
glasses = 0
#happy
happy = 0
#young and olg
young = 0
old = 0

counter = 0
start = time.perf_counter()
for i in os.listdir(path):
    file = open(path + "/" + i)


    data = json.load(file)
    #Combination of happy and smiling
    combination_happy = 0
    if data["df_emotion"] == "happy":
        combination_happy = (data["Smiling"] + (data["df_emotion_prob"] / 100)) / 2

    if data["Male"] >= 0.5:
        male += 1
    if data["Male"] < 0.5:
        female += 1
    if data["df_race"] == "white":
        white += 1
    if data["df_race"] == "indian":
        indian += 1
    if data["df_race"] == "asian":
        asian += 1
    if data["df_race"] == "middle eastern":
        middleeastern += 1
    if data["df_race"] == "black":
        black += 1
    if data["df_race"] == "latino hispanic":
        latino += 1
    if data["Eyeglasses"] > 0.5:
        glasses += 1
    if combination_happy > 0.5:
        happy += 1
    if data["Young"] >= 0.5:
        young += 1
    if data["Young"] < 0.5:
        old += 1


    file.close()
    counter += 1
stop = time.perf_counter()
final_1 = stop - start

total = male + female

#Gender results
print("Computational time: " + str(final_1))
print("males: " + str(male / total))
print("Female: " + str(female / total))

#Ethnicity results
print("White: " + str(white / total))
print("Black: " + str(black / total))
print("Indian: " + str(indian / total))
print("Asian: " + str(asian / total))
print("Middle eastern: " + str(middleeastern / total))
print("Latino: " + str(latino / total))

#plots
#ethnicities
x = ["Asian", "Indian", "Black", "White", "Middle eastern", "Latino"]
y = [asian, indian, black, white, middleeastern, latino]
#genders
x_gender = ["Male", "Female"]
y_gender = [male, female]
# bar plots
#figure = plt.figure(figsize=(10, 5))
# Pie plots
plt.pie(y, labels=x, autopct="%.1f%%")
plt.title("Share of ethnicities")
plt.show()
# https://www.w3schools.com/python/matplotlib_pie_charts.asp
plt.pie(y_gender, labels=x_gender, autopct="%.1f%%")
plt.title("Share of genders")
plt.show()
#glasses
x_glasses = ["Glasses","No glasses"]
y_glasses = [glasses, counter - glasses]
plt.pie(y_glasses, labels=x_glasses, autopct="%.1f%%")
plt.title("Share of people with glasses")
plt.show()
# https://www.w3schools.com/python/matplotlib_pie_charts.asp

#happy
x_happy = ["Happy people","Not happy people"]
y_happy = [happy, counter - happy]
plt.pie(y_happy, labels=x_happy, autopct="%.1f%%")
plt.title("Share of happy people")
plt.show()
# https://www.w3schools.com/python/matplotlib_pie_charts.asp

#age
x_age = ["young", "old"]
y_age = [young, old]
plt.pie(y_age, labels=x_age, autopct="%.1f%%")
plt.title("Share of young and old")
plt.show()
# https://www.w3schools.com/python/matplotlib_pie_charts.asp

