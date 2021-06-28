import numpy as np
from monk import BBox
import tensorflow as tf
from monk import Dataset
import json
import PIL
from monk.utils.s3.s3path import S3Path

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,json_paths, batch_size=10, dim=(128,128), n_channels=3,shuffle=True,damaged=False):
        
        
        
        self.shuffle = shuffle 
        self.dim = dim 
        self.batch_size = batch_size  
        self.n_channels = n_channels
        self.damaged=damaged
        
        jsons_data=[]
        
        for json_path in json_paths:
            with open(json_path) as f:
                json_data = json.load(f)
            jsons_data.append(json_data)
            
        self.filter_json(jsons_data,damaged)

        
        
    def filter_json(self,jsons_data,damaged):
        
        
        filtered_json =[]
        labels = []
        paths = []

        for json_data in jsons_data :
        
            for i in range(0,len(json_data)):
                
                if( not json_data[i]["path"] in paths):
                    

                    if json_data[i]["repair_action"]=='not_damaged':
                        labels.append(1)
                        filtered_json.append(json_data[i])
                        paths.append(json_data[i]["path"])
                    elif json_data[i]["repair_action"]!='not_damaged' and json_data[i]["repair_action"]!=None and (json_data[i]["label"]=='scratch' or json_data[i]["label"]=='dent' ) :
                        labels.append(-1)
                        filtered_json.append(json_data[i])
                        paths.append(json_data[i]["path"])
               
       
        self.labels = labels
        self.filtered_json=filtered_json
        self.list_IDs = np.arange(len(filtered_json)) 
        self.indexes = np.arange(len(filtered_json))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        

    
    def load_image(self,id):
        
        data = self.filtered_json[id]
        label = self.labels[id]
        
        
        if('s3:/monk-client-images/' in data["path"]):
            bucket = "monk-client-images"
            key = data["path"].replace("s3:/monk-client-images/","")
            s3 = S3Path(bucket,key)
            im = PIL.Image.open(s3.download())
        else:
            im = PIL.Image.open(data["path"])
        
        bbox =  data["part_bbox"]
        img_crop = im.crop(bbox)
        img_crop = img_crop.resize(self.dim)
        
        return(  ((((np.array(img_crop)/255)*2)-1)).astype(np.float32),label)
        #return(np.array(img_crop).astype(np.float32))
        
    def __len__(self):
       
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
    
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X,y = self.__data_generation(list_IDs_temp)

        return tf.convert_to_tensor(X,dtype=tf.float32),tf.convert_to_tensor(y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels),dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,],y[i] = self.load_image(self.indexes[ID])

            

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return tf.convert_to_tensor(X),tf.convert_to_tensor(y)

def get_generator(json_paths,batch_size,size,damaged=False,shuffle=True):
    
    generator = DataGenerator(json_paths,batch_size=batch_size,dim=(size,size),damaged=damaged,shuffle=shuffle)

    return(generator)