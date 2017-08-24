import os
import numpy as np

np.random.seed(0)

# imports 
import sys
import csv
from sklearn import svm

# variables
save_path = sys.argv[1]
train_path = sys.argv[2]
test_path = sys.argv[3]
output_path = sys.argv[4]

# load train, test list
with open(train_path) as f:
	train_list = [x.split('\t')[0] for x in f.read().splitlines()]
with open(train_path) as f:	
	train_label = [x.split('\t')[1] for x in f.read().splitlines()]
with open(test_path) as f:
	test_list = [x.split('\t')[0] for x in f.read().splitlines()]

# unique labels
unq_labels = list(set(train_label))

# generate one hot matrix
train_size = len(train_list)
test_size = len(test_list)
num_tags = len(unq_labels)

# shuffling training set
train_list = np.array(train_list)
train_label = np.array(train_label)
tmp1 = np.arange(train_size)
np.random.shuffle(tmp1)
train_list = train_list[tmp1]
train_label = train_label[tmp1]
train_list = train_list.tolist()
train_label = train_label.tolist()

train_list_to_label = dict(zip(train_list,train_label))

y_train = np.zeros((train_size,num_tags)) 

for sample_iter in range(train_size):
	for tag_iter in range(num_tags):
		if train_list_to_label[train_list[sample_iter]] == unq_labels[tag_iter]:
			y_train[sample_iter,tag_iter] = 1


# load 1 sample for measure feature_length
tmp_feature = np.load(save_path + train_list[0].replace('.wav','.npy'))
feature_length = len(tmp_feature)
print feature_length

# load encoded feature
x_train = np.zeros((train_size,feature_length))
x_test = np.zeros((test_size,feature_length))

for iter in range(0,train_size):
	file_path = save_path + train_list[iter].replace('.wav','.npy')
	x_train[iter] = np.load(file_path)

	if np.remainder(iter,1000) == 0:
		print iter
print iter+1
for iter in range(0,test_size):
	file_path = save_path + test_list[iter].replace('.wav','.npy')
	x_test[iter] = np.load(file_path)

	if np.remainder(iter,1000) == 0:
		print iter
print iter+1

# normalization
mean_value = np.mean(x_train)
std_value = np.std(x_train)

x_train -= mean_value
x_test -= mean_value
x_train /= std_value
x_test /= std_value

print 'mean value: ' + str(mean_value)
print 'std value: ' + str(std_value)
print 'Normalization done!'

# svm
clf = svm.SVC()
y_train = np.argmax(y_train,axis=1)
clf.fit(x_train,y_train)
y_test_tmp = clf.predict(x_test)

output = np.zeros((test_size,num_tags))
for sample_iter in range(test_size):
	for tag_iter in range(num_tags):
		if y_test_tmp[sample_iter] == tag_iter:
			output[sample_iter,tag_iter] = 1


print output.shape

# write result
# output_path, unq_labels
with open(output_path,'wb') as f:
	wr = csv.writer(f,quoting=csv.QUOTE_NONE,delimiter='\t')
	tag_index_list = np.argmax(output,axis=1)
	for file_iter in range(test_size):
		prints = [test_list[file_iter], unq_labels[tag_index_list[file_iter]]]
		wr.writerow(prints)
print 'write done'












