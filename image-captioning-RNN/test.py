from pickle import load
from numpy import argmax
from keras_preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
import numpy as np

base_model = VGG16(include_top=True)
feature_extract_pred_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

def extract_feature(model, file_name):
    img = load_img(file_name, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    fc2_features = model.predict(x)
    return fc2_features


caption_train_tokenizer = load(open('caption_train_tokenizer.pkl', 'rb'))
max_length = 33
pred_model = load_model('modelConcat_1_2.h5') 

def generate_caption(pred_model, caption_train_tokenizer, photo, max_length):
    in_text = '<START>'
    caption_text = list()
    for i in range(max_length):
           
            sequence = caption_train_tokenizer.texts_to_sequences([in_text])[0]
           
            sequence = pad_sequences([sequence], maxlen=max_length)
           
            model_softMax_output = pred_model.predict([photo,sequence], verbose=0)
           
            word_index = argmax(model_softMax_output)
           
            word = caption_train_tokenizer.index_word[word_index]
           
           
            if word is None:
                break
           
            in_text += ' ' + word
           
            if word != 'end':
                caption_text.append(word)
            if word == 'end':
                break
    return caption_text

caption_image_fileName = 'test.jpg'
photo = extract_feature(feature_extract_pred_model, caption_image_fileName)
caption = generate_caption(pred_model, caption_train_tokenizer, photo, max_length)
print(' '.join(caption))