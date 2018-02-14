import glob
import numpy as np
import pandas as pd
import keras
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

#####################
# helper functions
#####################
# define image preprocessor for use with keras vgg19
def image_preprocessor(image_path):
    image = image_utils.load_img(image_path, target_size=(224, 224))
    image = image_utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return(image)

#function to read in files housing all cat/dog breeds in imagenet labels
def read_dog_cat_labels(path):
    labs = list(open(path))
    labs = [item.split(',') for item in labs]
    labs = [item.strip().replace(' ','_') for sublist in labs for item in sublist]
    return(labs)
#------------------------------------------------

#####################
# prep images for classification
#####################
# define image paths to classify
image_files = glob.glob('images/*/*')

# preprocess images
image_list = [image_preprocessor(path) for path in image_files]
#------------------------------------------------

#####################
# load model and make predictions
#####################
# load vgg19 model pretrained with imagenet
model = keras.applications.VGG19(weights='imagenet')

# get model predictions
preds = [model.predict(image) for image in image_list]
preds = [list(decode_predictions(pred, top=1)[0][0]) for pred in preds]

# convert list of predictions to df and drop class name column
pred_df = pd.DataFrame(preds)
pred_df = pred_df.drop(0, 1)

#make names match names in R output for consistency
pred_df.columns = ['class_description', 'score']
#------------------------------------------------

#####################
# add dog/cat labels, add file name, sort by score
#####################
#read in breed labels
dog_labs = read_dog_cat_labels('data/dog_classes.txt')
cat_labs = read_dog_cat_labels('data/cat_classes.txt')

#create column for labeling dog breeds as dog and cat breeds as cat
pred_df['catdog'] = np.nan
pred_df.loc[pred_df.class_description.isin(dog_labs), 'catdog'] = 'Dog'
pred_df.loc[pred_df.class_description.isin(cat_labs), 'catdog'] = 'Cat'

#add column with image paths
pred_df['file_name'] = image_files
pred_df = pred_df.sort_values('score', ascending=False)
#------------------------------------------------

#####################
# print output
#####################
print(pred_df)

#   class_description     score catdog                               file_name
# 1         Chihuahua  0.777919    Dog      images/dogs/tonks_scary_sneeze.png
# 4             tabby  0.477750    Cat        images/cats/google_tabby_cat.jpg
# 3      Egyptian_cat  0.449422    Cat         images/cats/goober_lounging.jpg
# 5              chow  0.278078    Dog  images/ambiguous/tonks_jasper_bone.jpg
# 2              lynx  0.147593    Cat             images/cats/lilly_perch.jpg
# 0             boxer  0.107166    Dog              images/dogs/tonks_beer.jpg
#------------------------------------------------
