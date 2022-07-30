# Kedi-Kopek-Tahmini-cat-dog-prediction-
Tensorflow,Keras ile verilen resmin kedi mi veya köpek mi olduğu tahminini yapan program.Verileri Kaggle'den aldım.

Eğer güçlü bir GPU'nuz yok ise Google Colab'ta eğitimi gerçekleştirmenizi tavsiye ederim.

Colab'a Google Drive'nizi bağlayın.Verisetimizi yüklemek aşağıdaki kod ile gerçekleşiyor.

```Python
from google.colab import drive
drive.mount('dataset')
```
Kütüphaneleri ve modülleri yükleme kısmı.
```Python
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
```
Eğitim verilerini hazırlama kısmı.Burada resimlerin yeniden ölçeklendirip aynı fotoğraftan farklı açılardan da öğrenmesini sağlıyoruz.Örnek olarak horizontal_flip ile resimleri yatay çevirir ve oradanda öğrenir.
```Python
trainset_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        )
train_set = trainset_datagen.flow_from_directory('../content/dataset/MyDrive/dataset/training_set',
            target_size=(64, 64),  
            batch_size=32,
            class_mode='binary')
```
Test verilerini hazırlama kısmı, eğitim ile aynı olaylar gerçekleştiriliyor.
```Python
testset_datagen = ImageDataGenerator(rescale=1./255)
test_set = testset_datagen.flow_from_directory(
            '../content/dataset/MyDrive/dataset/test_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')
```
Burada CNN(Evrişimsel sinir ağı) modeli oluşturulacak.
```Python
cnn = tf.keras.models.Sequential()
```
Burada Evrişim Katmanı (CONV) layers.Conv2D kodunda bunu gerçekleştiriyor.Bu katmanda resmin özellikleri saptamak için kullanılır.Pooling katmanı, layers.MaxPooling2D kodunda bunu gerçekleştiriyor.Bu katmanın görevi, gösterimin kayma boyutunu ve ağ içindeki parametreleri ve hesaplama sayısını azaltmak içindir.Birçok Pooling işlemleri vardır, fakat burada maxpooling kullanıldı.
```Python
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3])) 
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2,strides=2))
```
```Python
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2,strides=2))
```
Burada son olarak Flattening katmanı var.Bu katmanın görevi basitçe, son ve en önemli katman olan Fully Connected Layer’ın girişindeki verileri hazırlamaktır.
```Python
cnn.add(tf.keras.layers.Flatten())
```
```Python

```
