# Denoising-AutoEncoder
# Image Denoising using AutoEncoder

In this project, we implemented an image denoising method which can be generally used
in all kinds of noisy images. We achieved denoising process by adding Gaussian noise to
raw images and then feed them into AutoEncoder to learn its core representations(raw
images itself or high-level representations).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install following required packages.

```bash
pip install tensorflow==1.15.0
```

## Import Packages

```python
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Input
from keras.models import Model

```

## Loading the training and test dataset

```python
(x_train,_),(x_test,_) = mnist.load_data()
```

## Adding noise to the data
```python
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0,scale=1.0,size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0,scale=1.0,size=x_test.shape)
```


## Encoder Part 
```python
input_img = Input(shape=(28,28,1))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)


```

## Decoder part
```python 
decoder_conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')
decoder_upsamp1 = UpSampling2D((2, 2))
decoder_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')
decoder_upsamp2 = UpSampling2D((2, 2))
decoder_conv3 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')

x = decoder_conv1(encoded)
x = decoder_upsamp1(x)
x = decoder_conv2(x)
x = decoder_upsamp2(x)
decoder_output = decoder_conv3(x)
```
## AutoEncoder Model
```python
#Creating autoencoder model
autoencoder = Model(input_img, decoder_output)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```

## Model Usage
```python
# Check the trained model performance on test set 
encoder_imgs = encoder.predict(x_test_noisy)
decoder_imgs = decoder.predict(encoder_imgs)

```

## Training the model
```python 
# Training the model on the dataset
autoencoder_train=autoencoder.fit(x_train_noisy, x_train,
                epochs=15,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))


```

Training and Validation Loss:
![Training and Validation Loss](https://github.com/ajayiiti/Denoising-AutoEncoder/blob/master/trainingAndValidationLoss.png)



Final Result:
![Denoised Images](https://github.com/ajayiiti/Denoising-AutoEncoder/blob/master/DenoisedImage.png)



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
