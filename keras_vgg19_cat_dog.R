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

#   class_description     score catdog                              file_name
# 6         Chihuahua 0.7779192    Dog     images/dogs/tonks_scary_sneeze.png
# 3             tabby 0.4777499    Cat       images/cats/google_tabby_cat.jpg
# 2      Egyptian_cat 0.4494216    Cat        images/cats/goober_lounging.jpg
# 1              chow 0.2780781    Dog images/ambiguous/tonks_jasper_bone.jpg
# 4              lynx 0.1475925    Cat            images/cats/lilly_perch.jpg
# 5             boxer 0.1071659    Dog             images/dogs/tonks_beer.jpg
#------------------------------------------------

#####################
# write labels on images and save to file
# this part is not included in the mirrored python script but 
# can be accomplished with packages such as opencv (see: https://www.pyimagesearch.com/2016/08/10/imagenet-classification-with-python-and-keras/)
# or with PIL (see: https://stackoverflow.com/questions/16373425/add-text-on-image-using-pil)
#####################

#helper function to write labels on images
write_class_results = function(file_path_in, file_path_out, label, score) {
  #get file ext
  out_ext = tolower(tools::file_ext(file_path_out))
  
  #read image
  img = magick::image_read(file_path_in)
  
  #resize
  img = magick::image_resize(img, magick::geometry_size_pixels(400, 400))
  #write label: score in top left
  img = magick::image_annotate(img, sprintf('%s: %.4f', label, score),
                               size = 20, color = '#00ff00')
  
  #write out
  magick::image_write(img, path = file_path_out, format = out_ext)
}

for (i in 1:nrow(pred_df)) {
  #print status
  cat('writing', i, 'of', nrow(pred_df), '\n')
  
  #get data for ith row
  row_i = pred_df[i, ]
  
  #create out file name in results dir
  out_file_name = paste0('results/', basename(row_i$file_name))

  #call write func
  write_class_results(file_path_in = row_i$file_name, 
                      file_path_out = out_file_name,
                      label = row_i$catdog, 
                      score = row_i$score)
}
#------------------------------------------------