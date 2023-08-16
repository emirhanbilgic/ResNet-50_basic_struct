import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os

dataset_path = os.listdir('rooms_dataset')

room_types = os.listdir('rooms_dataset')
print (room_types)  #what kinds of rooms are in this dataset

print("Types of rooms found: ", len(room_types)) 

#########

rooms = []

for item in room_types:
 #get all the file names
 all_rooms = os.listdir('rooms_dataset' + '/' +item)
 #print(all_shoes)

 # Add them to the list
 for room in all_rooms:
    rooms.append((item, str('rooms_dataset' + '/' +item) + '/' + room))
    #print(rooms[:1])

rooms

###########

# Build a dataframe        
rooms_df = pd.DataFrame(data=rooms, columns=['room type', 'image'])
print(rooms_df.head())
#print(rooms_df.tail())

###########

# Let's check how many samples for each category are present
print("Total number of rooms in the dataset: ", len(rooms_df))

room_count = rooms_df['room type'].value_counts()

print("rooms in each category: ")
print(room_count)

###########

import cv2
path = 'rooms_dataset/'


im_size = 224

images = []
labels = []

for i in room_types:
    data_path = path + str(i)  # entered in 1st folder and then 2nd folder and then 3rd folder
    filenames = [i for i in os.listdir(data_path) ]
   # print(filenames)  # will get the names of all images
    for f in filenames:
        img = cv2.imread(data_path + '/' + f)  # reading that image as array
        #print(img)  # will get the image as an array
        img = cv2.resize(img, (im_size, im_size))
        images.append(img)
        labels.append(i)
labels

###########

images = np.array(images)
images.shape

###########

#load ResNet50 model without the top classification layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(im_size, im_size, 3))

#add custom layers for your classification task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(room_types), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

#preprocess your images
images = images / 255.0  # Normalize to [0,1]

#convert labels to integers and then to one-hot encoded labels
le = LabelEncoder()
integer_encoded_labels = le.fit_transform(labels)
onehot_labels = to_categorical(integer_encoded_labels)

#compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(images, onehot_labels, epochs=10, batch_size=32, validation_split=0.1)


