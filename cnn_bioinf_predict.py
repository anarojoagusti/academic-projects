## Convolutional Neural Network - Ana1
"""
Created on Sat Mar 31 20:00:30 2018

@author: anaro
"""

## Part 3. Writing predictions
file=open(pathwork + 'PREDICT_DATA\\Labels.csv','r')
fileLabels=csv.reader(file)
labels=list()
for line in fileLabels:
    labels=line
file.close()
pred_labels=[int(i) for i in labels]
    
filesImages=os.listdir()
del filesImages[filesImages.index('Labels.csv')]
labelsCNN=list(range(len(filesImages)))
for i in range(len(filesImages)):
    test_image = image.load_img(filesImages[i], target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    indices=train_set.class_indices
    labelsCNN[i]=result.argmax()
       
accuracy=accuracy_score(pred_labels, labelsCNN)
recall=recall_score(pred_labels,labelsCNN,average='weighted')   
precision=precision_score(pred_labels,labelsCNN,average='weighted')   
    
#if result[0][0] == 0:
#prediction = 'flower'
#if result[0][0] == 1:
#prediction = 'fruit'
#if result[0][0] == 2:
#prediction = 'leaf'
#if result[0][0] == 3:
#prediction = 'trichoma'
