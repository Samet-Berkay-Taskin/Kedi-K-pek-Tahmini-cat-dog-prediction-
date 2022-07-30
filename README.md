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
Burada convolutional(evrişim) Katmanı, kod olarak layers.Conv2D bunu gerçekleştiriyor.Bu katmanda resmin özellikleri saptamak için kullanılır.Pooling (havuzlama) katmanı bu katmanın görevi, gösterimin kayma boyutunu ve ağ içindeki parametreleri ve hesaplama sayısını azaltmak içindir.Birçok Pooling işlemi vardır, fakat burada maxpooling kullanıldı.Kod olarak layers.MaxPooling2D bunu gerçekleştiriyor.
```Python
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3])) 
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2,strides=2))
```
```Python
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2,strides=2))
```
Burada son olarak Flattening katmanı var.Bu katmanın görevi basitçe, son ve en önemli katman olan fully connected katmanı’nın (sinir ağına) girişindeki verileri hazırlamaktır.Sinir ağları, giriş verilerini tek boyutlu bir diziden alır. Bu da sinir ağındaki veriler ise convolutional(Evrişim) ve pooling(Havuzlama) katmanından gelen matrixlerin tek boyutlu diziye çevrilmiş halidir.
```Python
cnn.add(tf.keras.layers.Flatten())
```
Şimdi ise fully connected katmanına verileri flattening işleminden alır ve Sinir ağı yoluyla öğrenme işlemini geçekleştirir.
Bu katman başlı başına bir sinir ağıdır.
```Python
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid')) # Burası çıkış katmanı
```
Eğitime başlasın.Burası biraz uzun sürebilir.
```Python
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) 
cnn.fit(x=train_set, validation_data=test_set, epochs=25)
```
Artık verilen resmin tahmin kısmına başlayabiliriz.
```Python
from keras.preprocessing import image
```
Sigmoid fonksiyonunda değerler 0 ile 1 arasındadır.Dolayısıyla verilen resmin sonucu eğer 1'e yakın ise köpek 0'a yakın ise kedi sonucunu döndürüyor.
Şimdiki göstereceğim kodda elimizdeki dosyadaki fotoğrafları istediğiniz gibi tahmin ettirtebilirsiniz. 
test_foto1 = image.load_img('../content/dataset/MyDrive/dataset/single_prediction/cat2 <<<<<------ burayı cat1 cat5 veya dog1 dog 3 yapabilirsiniz single_prediction dosyasına kendiniz kedi köpek resmi koyup onlarıda deneyebilirsiniz.
```Python
test_foto1 = image.load_img('../content/dataset/MyDrive/dataset/single_prediction/cat2.jpg', target_size=(64,64))
test_foto1 = image.img_to_array(test_foto1)
test_foto1 = test_foto1/255
test_foto1 = np.expand_dims(test_foto1, axis=0)
sonuc1 = cnn.predict(test_foto1)
train_set.class_indices 
if sonuc1[0][0] > 0.5:
    Prediction = 'Dog'
else:
    Prediction = 'Cat'
print(Prediction)
```
Burada aslında programımız tamamlandı fakat eğer bu eğitilmiş programı alıp Pyhcarm veya kendiniz herhangi bir Python IDE'de çalıştırmak isterseniz eğtim bittikten sonrar bu kodla kaydedebilirsiniz.HDF5 formatında kaydeder.Bunu colabte eğitimden sonra yapın.
```Python
cnn.save('kedi-kopek.h5') 
```
Eğitilmiş programı bu kod ile çalıştırabilirsin.Bu kodu istediğniz Python IDE'sinde kullanabilirsiniz.
```Python
from keras.models import load_model

new_model = load_model('kedi-kopek.h5')
new_model.summary()
```
