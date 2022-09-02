# Image-Colorization-System

Image colorization is the process of assigning colors to a grayscale image to make it more aesthetically appealing and perceptually meaningful. These are recognized as sophisticated tasks than often require prior knowledge of image content and manual adjustments to achieve artifact-free quality.

![image](https://user-images.githubusercontent.com/64821137/188227907-666ff003-9e30-40b9-b3a7-b5df11df29b0.png)

## Approach

We train a CNN to map from a grayscale input to a distribution over quantized color value outputs using the architecture shown. 

![image](https://user-images.githubusercontent.com/64821137/188229119-fba122d4-5041-4a2b-a9b1-262a5a5efff7.png)

To train the network start with the ImageNet dataset converting all images from the RGB color space to the `Lab color space.`

Similar to the RGB color space, `the Lab color space has three channels. But unlike the RGB color space, Lab encodes color information differently:`

* The L channel encodes lightness intensity only
* The a channel encodes green-red.
* And the b channel encodes blue-yellow

Since the L channel encodes only the intensity, we can use the `L channel as our grayscale input to the network.`

From there the network `must learn to predict the a and b channels.` Given the input L channel and the predicted ab channels we can then form our final output image.

## Results

```
pip install -r requirements.txt

python app.py -i imgs/1.jpg
```

![image](https://user-images.githubusercontent.com/64821137/188229793-2f3d5851-51f2-4611-ae49-0d87597513f0.png)

