from pickle import dump
from pickle import load
from keras.applications.vgg16 import VGG16
from keras_preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
from keras.preprocessing.text import Tokenizer
from collections import Counter
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import add
from keras.utils import plot_model
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
from os import listdir
base_model = VGG16(include_top=True)
base_model.summary()

model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
model.summary()
try:
	features = load(open('features.pkl', 'rb'))
except Exception as e:
    
	features = dict()
	for file in listdir('Flickr8k_Dataset/Flicker8k_Dataset'):
		img_path = 'Flickr8k_Dataset/Flicker8k_Dataset/' + file
		img = load_img(img_path, target_size=(224, 224))
		x = img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		fc2_features = model.predict(x)
		
		name_id = file.split('.')[0]
		features[name_id] = fc2_features

	dump(features, open('features.pkl', 'wb'))

def load_data_set_ids(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    
    dataset = list()
    for image_id in text.split('\n'):
        if len(image_id) < 1:
            continue
            
        dataset.append(image_id)
    
    return set(dataset)

training_set = load_data_set_ids('Flickr8k_text/Flickr_8k.trainImages.txt')
dev_set = load_data_set_ids('Flickr8k_text/Flickr_8k.devImages.txt')
test_set = load_data_set_ids('Flickr8k_text/Flickr_8k.testImages.txt')

import string
filename = 'Flickr8k_text/Flickr8k.token.txt'
file = open(filename, 'r')
token_text = file.read()
file.close()

translator = str.maketrans("", "", string.punctuation)
image_captions = dict()
image_captions_train = dict()
image_captions_dev = dict()
image_captions_test = dict()
image_captions_other = dict()
corpus = list()
corpus.extend(['<START>', '<END>', '<UNK>'])

max_imageCap_len = 0

for line in token_text.split('\n'):
    tokens = line.split(' ')
    if len(line) < 2:
        continue
    image_id, image_cap = tokens[0], tokens[1:]
    image_id = image_id.split('#')[0]
    image_cap = ' '.join(image_cap)

    image_cap = image_cap.lower()
    image_cap = image_cap.translate(translator)
    
    image_cap = image_cap.split(' ')
    image_cap = [w for w in image_cap if w.isalpha()]
    image_cap = [w for w in image_cap if len(w)>1]
    image_cap = '<START> ' + ' '.join(image_cap) + ' <END>'
    
   
    if len(image_cap.split()) > max_imageCap_len:
        max_imageCap_len = len(image_cap.split())
    
   
    if image_id not in image_captions:
        image_captions[image_id] = list()
    image_captions[image_id].append(image_cap)
    
   
    if image_id in training_set:
        if image_id not in image_captions_train:
            image_captions_train[image_id] = list()
        image_captions_train[image_id].append(image_cap)
        corpus.extend(image_cap.split())
        
    elif image_id in dev_set:
        if image_id not in image_captions_dev:
            image_captions_dev[image_id] = list()
        image_captions_dev[image_id].append(image_cap)
        
    elif image_id in test_set:
        if image_id not in image_captions_test:
            image_captions_test[image_id] = list()
        image_captions_test[image_id].append(image_cap)
    else:
        if image_id not in image_captions_other:
            image_captions_other[image_id] = list()
        image_captions_other[image_id].append(image_cap)

caption_train_tokenizer = Tokenizer()
caption_train_tokenizer.fit_on_texts(corpus)
    
fid = open("image_captions.pkl","wb")
dump(image_captions, fid)
fid.close()

fid = open("image_captions_train.pkl","wb")
dump(image_captions_train, fid)
fid.close()

fid = open("image_captions_dev.pkl","wb")
dump(image_captions_dev, fid)
fid.close()

fid = open("image_captions_test.pkl","wb")
dump(image_captions_test, fid)
fid.close()

fid = open("image_captions_other.pkl","wb")
dump(image_captions_other, fid)
fid.close()

fid = open("caption_train_tokenizer.pkl","wb")
dump(caption_train_tokenizer, fid)
fid.close()

fid = open("corpus.pkl","wb")
dump(corpus, fid)
fid.close()

corpus_count=Counter(corpus)
fid = open("corpus_count.pkl","wb")
dump(corpus_count, fid)
fid.close()

print("size of data =", len(image_captions), "size of training data =", len(image_captions_train), "size of dev data =", len(image_captions_dev), "size of test data =", len(image_captions_test), "size of unused data =", len(image_captions_other))
print("maximum image caption length =",max_imageCap_len)

embeddings_index = dict()
fid = open('glove.6B.50d.txt' ,encoding="utf8")
for line in fid:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
fid.close()

EMBEDDING_DIM = 50
word_index = caption_train_tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

for word, idx in word_index.items():
    embed_vector = embeddings_index.get(word)
    if embed_vector is not None:
       
        embedding_matrix[idx] = embed_vector
        
fid = open("embedding_matrix.pkl","wb")
dump(embedding_matrix, fid)
fid.close()

def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    X1, X2, y = list(), list(), list()
   
    for desc in desc_list:
       
        seq = tokenizer.texts_to_sequences([desc])[0]
       
        for i in range(1, len(seq)):
           
            in_seq, out_seq = seq[:i], seq[i]
           
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
           
           
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
           
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(np.squeeze(X1)), np.array(X2), np.array(y)

def data_generator(descriptions, photos, tokenizer, max_length, batch_size, vocab_size):
   
    current_batch_size=0
    while 1:
        for key, desc_list in descriptions.items():
           
            if current_batch_size == 0:
                X1, X2, Y = list(), list(), list()
            
            imageFeature_id = key.split('.')[0]
            photo = photos[imageFeature_id][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
           
            X1.extend(in_img)
            X2.extend(in_seq)
            Y.extend(out_word)
            current_batch_size += 1
            if current_batch_size == batch_size:
                current_batch_size = 0
                yield [[np.array(X1), np.array(X2)], np.array(Y)]
from pickle import load
fid = open('features.pkl', 'rb')
image_features = load(fid)
fid.close()

caption_max_length = 33
batch_size = 1
vocab_size = 7057
generator = data_generator(image_captions_train, image_features, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
inputs, outputs = next(generator)
print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)

from keras.layers import concatenate
def define_model_concat(vocab_size, max_length, embedding_matrix):
   
    inputs1 = Input(shape=(4096,))
    image_feature = Dropout(0.5)(inputs1)
    image_feature = Dense(256, activation='relu')(image_feature)
   
    inputs2 = Input(shape=(max_length,))
    language_feature = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=max_length, trainable=False)(inputs2)
   
    language_feature = Dropout(0.5)(language_feature)
    language_feature = LSTM(256)(language_feature)
   
    output = concatenate([image_feature, language_feature])
    output = Dense(256, activation='relu')(output)
    output = Dense(vocab_size, activation='softmax')(output)
   
    model = Model(inputs=[inputs1, inputs2], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
   
    print(model.summary())
    plot_model(model, to_file='model_concat.png', show_shapes=True)
    return model

fid = open("embedding_matrix.pkl","rb")
embedding_matrix = load(fid)
fid.close()

caption_max_length = 33
vocab_size = 7506
post_rnn_model_concat = define_model_concat(vocab_size, caption_max_length, embedding_matrix)

fid = open("features.pkl","rb")
image_features = load(fid)
fid.close()

fid = open("caption_train_tokenizer.pkl","rb")
caption_train_tokenizer = load(fid)
fid.close()

fid = open("image_captions_train.pkl","rb")
image_captions_train = load(fid)
fid.close()

fid = open("image_captions_dev.pkl","rb")
image_captions_dev = load(fid)
fid.close()

caption_max_length = 33
batch_size = 100
vocab_size = 7506
batch_size = 6
steps = len(image_captions_train)
steps_per_epoch = np.floor(steps/batch_size)

epochs = 3

for i in range(epochs):

	generator = data_generator(image_captions_train, image_features, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)

	post_rnn_model_concat_hist=post_rnn_model_concat.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)

	post_rnn_model_concat.save('modelConcat_1_' + str(i) + '.h5')

