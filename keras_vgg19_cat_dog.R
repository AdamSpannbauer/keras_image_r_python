library(keras)

#####################
# helper functions
#####################
# define image preprocessor for use with keras vgg19
image_preprocessor = function(image_path) {
  image_load(image_path, target_size = c(224,224)) %>% 
    image_to_array() %>% 
    array_reshape(c(1, dim(.))) %>% 
    imagenet_preprocess_input()
}

#function to read in files housing all cat/dog breeds in imagenet labels
read_dog_cat_labels = function(path) {
  labs = readLines(path)
  labs = trimws(unlist(strsplit(labs, ',')))
  labs = gsub('\\s+', '_', labs)
  return(labs)
}
#------------------------------------------------

#####################
# prep images for classification
#####################
# define image paths to classify
image_paths = list.files('images', recursive = TRUE, full.names = TRUE)

# preprocess images
image_list = lapply(image_paths, image_preprocessor)
#------------------------------------------------

#####################
# load model and make predictions
#####################
# load vgg19 model pretrained with imagenet
model = application_vgg19()

# get model prediction
preds = lapply(image_list, function(i) {
  imagenet_decode_predictions(predict(model, i), top = 1)[[1]]
})

# convert list of predictions to df and drop class name column
pred_df = do.call(rbind, preds)
pred_df$class_name = NULL
#------------------------------------------------

#####################
# add dog/cat labels, add file name, sort by score
#####################
#read in breed labels
dog_labs = read_dog_cat_labels('data/dog_classes.txt')
cat_labs = read_dog_cat_labels('data/cat_classes.txt')

#create column for labeling dog breeds as dog and cat breeds as cat
pred_df$catdog = NA
pred_df$catdog[pred_df$class_description %in% dog_labs] = 'Dog'
pred_df$catdog[pred_df$class_description %in% cat_labs] = 'Cat'

#add column with image paths
pred_df$file_name = unlist(image_paths)
pred_df = pred_df[order(-pred_df$score),]
#------------------------------------------------

#####################
# print output
#####################
print(pred_df)

#   class_description     score catdog                                  file_name
# 7         Chihuahua 0.9420919    Dog        images/dogs/tonks_proposal_sign.jpg
# 6         Chihuahua 0.7851309    Dog images/dogs/tonks_jasper_announce_sign.jpg
# 5          malinois 0.5802087    Dog        images/dogs/tonks_announce_sign.jpg
# 2             tabby 0.4777499    Cat           images/cats/google_tabby_cat.jpg
# 1      Egyptian_cat 0.4494216    Cat            images/cats/goober_lounging.jpg
# 3    Scotch_terrier 0.3038349    Dog       images/cats/jasper_announce_sign.jpg
# 4              lynx 0.1475925    Cat                images/cats/lilly_perch.jpg
#------------------------------------------------
