import numpy as np
import gzip
import numpy as np
import pandas as pd
from time import time
from requests import get
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score

from dbn.tensorflow import SupervisedDBNClassification
#from dbn import SupervisedDBNClassification


def read_mnist(images_path: str, labels_path: str):
    with gzip.open(labels_path, 'rb') as labelsFile:
        labels = np.frombuffer(labelsFile.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path,'rb') as imagesFile:
        length = len(labels)
        # Load flat 28x28 px images (784 px), and convert them to 28x28 px
        features = np.frombuffer(imagesFile.read(), dtype=np.uint8, offset=16) \
                        .reshape(length, 784) \
                        .reshape(length, 28,28,1)
        
    return features, labels

train = {}
test = {}

train['features'], train['labels'] = read_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
test['features'], test['labels'] = read_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

train['features'] = (train['features']).astype(np.float32) #data scaling
train['features']  = train['features'] /255


test['features'] = (test['features']).astype(np.float32)
test['features']  = test['features'] /255

print('# of training images:', train['features'].shape)
print(train['labels'].shape)
print('# of test images:', test['features'].shape)


X_train, Y_train = train['features'], train['labels']
print(X_train.shape)
print(Y_train.shape)
X_train = X_train.reshape((60000,-1)) #reshape para entrada 
print(X_train.shape)
print(Y_train.shape)


X_test, Y_test = test['features'], test['labels']

X_test = X_test.reshape((10000,-1))
print(X_test.shape)
print(Y_test.shape)


classifier = SupervisedDBNClassification(hidden_layers_structure=[512, 512],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.01,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=1000,
                                         batch_size=256,
                                         activation_function='relu',
                                         dropout_p=0.2)
inicio_train = time.time()

classifier.fit(X_train, Y_train)

fim_train = time.time()



classifier.save('model_CPU.pkl')


classifier = SupervisedDBNClassification.load('model_CPU.pkl')


inicio_pred = time.time()
Y_pred = classifier.predict(X_test)
fim_pred = time.time()

print('Tempo Pred(s): ', fim_pred - inicio_pred)
print('Tempo Train(s): ', fim_train - inicio_train)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))
