# Image-Colorization-System

Image colorization is the process of assigning colors to a grayscale image to make it more aesthetically appealing and perceptually meaningful. These are recognized as sophisticated tasks than often require prior knowledge of image content and manual adjustments to achieve artifact-free quality.

![image](https://user-images.githubusercontent.com/64821137/188227907-666ff003-9e30-40b9-b3a7-b5df11df29b0.png)

The system is implemented as a feed-forward pass in a CNN at test time and is trained on over a million color images. Evaluate the algorithm using a colorization Turing test asking human participants to choose between a generated and ground truth color image.

## Approach

We train a CNN to map from a grayscale input to a distribution over quantized color value outputs using the architecture shown. 

![image](https://user-images.githubusercontent.com/64821137/188229119-fba122d4-5041-4a2b-a9b1-262a5a5efff7.png)

## Results

```
pip install -r requirements.txt
python app.py -i imgs/1.jpg
```

![image](https://user-images.githubusercontent.com/64821137/188229793-2f3d5851-51f2-4611-ae49-0d87597513f0.png)
