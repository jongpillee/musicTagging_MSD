import os
import numpy as np
import time

from keras.optimizers import SGD
from keras.models import model_from_json,Model
from keras import backend as K
from keras.callbacks import Callback,ModelCheckpoint,EarlyStopping
from keras.layers import Input
from keras.layers.core import Dense

import sys
import librosa
import json

# load model
model_path = './models/'

architecture_name = model_path + 'architecture_msdTag.json'
weight_name = model_path + 'weight_msdTag.hdf5'

nst = 0
partition = 1

save_path = sys.argv[1]
train_arg = sys.argv[2]

fs = 22050

# read 50 tag labels
with open('./50tagList.txt') as f:
	tag_list = [x for x in f.read().splitlines()]
print tag_list

def load_melspec(file_name_from,num_segment,sample_length):
	#file_name = file_name_from.replace('.wav','.au')
	file_name = file_name_from
	
	tmp,sr = librosa.load(file_name,sr=fs,mono=True)
	tmp = tmp.astype(np.float32)
	
	y_length = len(tmp)

	tmp_segmentized = np.zeros((num_segment,sample_length,1))
	for iter2 in range(0,num_segment):
		
		hopping = (y_length-sample_length)/(num_segment-1)
		count_tmp = 0
		if hopping < 0:
			if count_tmp == 0:
				tmp_tmp = np.repeat(tmp,10)
				count_tmp += 1
			y_length_tmp = len(tmp_tmp)
			hopping = (y_length_tmp - sample_length)/(num_segment-1)
			tmp_segmentized[iter2,:,0] = tmp_tmp[iter2*hopping:iter2*hopping+sample_length]
		else:
			tmp_segmentized[iter2,:,0] = tmp[iter2*hopping:iter2*hopping+sample_length]

	return tmp_segmentized


# load data
with open(train_arg) as f:
	train_list = [x.split('\t')[0] for x in f.read().splitlines()]

print len(train_list)
all_list = train_list
print len(all_list)

model = model_from_json(open(architecture_name).read())
model.load_weights(weight_name)
print 'model loaded!!!'


# compile & optimizer
sgd = SGD(lr=0.001,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

# print model summary
model.summary()

sample_length = model.input_shape[1]
print sample_length

num_segment = int(22050*30/sample_length)+1
print 'Number of segments per song: ' + str(num_segment)


# define activation layer
layer_dict = dict([(layer.name,layer) for layer in model.layers[1:]])

# msd doesn't have dropout so +1 for capturing last hidden layer
activation_layer = 'dense_1'
print activation_layer

layer_output = layer_dict[activation_layer].output
get_last_output = K.function([model.layers[0].input, K.learning_phase()], [layer_output])

# encoding
all_size = len(all_list)
for iter2 in range(int(nst*all_size/partition),int((nst+1)*all_size/partition)):
	# check existence
	save_name = save_path + '/' + all_list[iter2].replace('.wav','.json')
	
	if not os.path.exists(os.path.dirname(save_name)):
		os.makedirs(os.path.dirname(save_name))
	
	if os.path.isfile(save_name) == 1:
		print iter2, save_name + '_file_exist!!!!!!!'
		continue

	# load melgram
	x_sample_tmp = load_melspec(all_list[iter2],num_segment,sample_length)
	print x_sample_tmp.shape

	# prediction
	weight = get_last_output([x_sample_tmp,0])[0]
	avgpooled = np.average(weight,axis=0)
	print avgpooled.shape,iter2

	# generate json object
	obj = {}
	obj['file_name'] = save_name
	obj['prediction_msd'] = {}
	for tag_iter in range(50):
		obj['prediction_msd'][tag_list[tag_iter]] = str(avgpooled[tag_iter])
		
	with open(save_name,'w') as outfile:
		json.dump(obj,outfile)



