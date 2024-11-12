import pandas as pd
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os.path
import glob




df_true = pd.DataFrame(columns = ['file_name','humans'])
df_false = pd.DataFrame(columns = ['file_name','humans'])

df_true['file_name'] = glob.glob(r'/Users/krish/ljmu/1.data/afo/tiledv2/test/ts/*.jpg')
df_true['humans'] = 1

df_false['file_name'] = glob.glob(r'/Users/krish/ljmu/1.data/afo/tiledv2/test/false/*.jpg')
df_false['humans'] = 0

tiled_test = df_true.append(df_false)

tiled_test['humans'] = tiled_test['humans'].apply(lambda x: str(x))

IMG_SIZE = 224

resize_and_rescale = tf.keras.Sequential([layers.Resizing(IMG_SIZE, IMG_SIZE),layers.Rescaling(1./255)])

def model_imp(model_path):
   model = tf.keras.models.load_model(model_path)


def predicter(file_path):
    
    #print(file_path)
    im = Image.open(file_path)
    test_input = np.array(im, dtype=np.uint8)
    if len(test_input.shape) == 3:
        test_input = resize_and_rescale(test_input)
        out = model(tf.reshape(test_input,(1,224,224,3)))

        return np.array(out)[0][0] 
    else:
        return 'out_of_shape'
    

def prediction_csv(model_path):
    
    if os.path.exists(model_path+'test_prediction.csv'):
        tiled_test.read_csv(model_path+'test_prediction.csv')
        y_true = tiled_test.iloc[:]['humans'].values
        y_pred = tiled_test.iloc[:]['prediction'].values

    else:    
        tiled_test['prediction'] = tiled_test.file_name.apply(predicter)
        tiled_test['humans'] = tiled_test.humans.apply(lambda x:int(x)) 
        tiled_test.to_csv(model_path+'test_prediction.csv')
        y_true = tiled_test.iloc[:]['humans'].values
        y_pred = tiled_test.iloc[:]['prediction'].values

    return y_pred,y_true