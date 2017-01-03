# Style Transfer

A TensorFlow implementation of style transfer based on the paper [A Neural Algortihm of Artistic Style](https://arxiv.org/pdf/1508.06576v2.pdf) by Gatys et. al.

## Algorithm

See my related blog post(link) for an overview of the style transfer algorithm.

The total loss used is the weighted sum of the style loss, the content loss and a total variation loss. This third component is not specfically mentioned in the original paper but leads to more cohesive images being generated.

## Requirements

* [Python 2.7](https://www.python.org/download/releases/2.7/)
* [TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup#download-and-setup)
* [SciPy & NumPy](http://scipy.org/install.html)
* Download the [pre-trained VGG network](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat) and place it in the top level of the repository (~500MB)

## Running the code

```python style_transfer.py --content <content image> --style <style image> --output <output image path>```

The algorithm will run with the following settings:

```python 
ITERATIONS = 1000    # override with --iterations argument
LEARNING_RATE = 1e1  # override with  --learning-rate argument
CONTENT_WEIGHT = 5e1 # override with --content-weight argument
STYLE_WEIGHT = 1e2   # override with --style-weight argument
TV_WEIGHT = 1e2      # override with --tv-weight argument
```

By default the algorithm will start with random noise image and optimise it to produce a style transferred image. To start with a particular image (for example the content image) run with the `--initial <initial image>` argument
    


## Results




## Acknowledgements

This code was inspired by an existing TensorFlow [implementation by Anish Athalye](https://github.com/anishathalye/neural-style), and I have re-used his VGG network code here.

## License

Released under GPLv3.
