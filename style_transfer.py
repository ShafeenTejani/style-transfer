# Copyright (c) 2016-2017 Shafeen Tejani. Released under GPLv3.

import vgg_network

import tensorflow as tf
import numpy as np

from sys import stdout
from functools import reduce


class StyleTransfer:
    CONTENT_LAYER = 'relu4_2'
    STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

    def __init__(self, vgg_path, content,
                style, content_weight,
                style_weight, tv_weight,
                initial,
                device):
        with tf.device(device):
            self.vgg = vgg_network.VGG(vgg_path)
            self.content = content
            self.style = style
            self.image = self._get_initial_image_or_random(initial)

            loss_calculator = LossCalculator(self.vgg, self.image);

            self.content_loss = loss_calculator.content_loss(content,
                                                             self.CONTENT_LAYER,
                                                             content_weight)

            self.style_loss = loss_calculator.style_loss(style,
                                                         self.STYLE_LAYERS,
                                                         style_weight)

            self.total_variation_loss = loss_calculator.tv_loss(self.image,
                                                                self.content.shape,
                                                                tv_weight)

            self.loss = self.content_loss + self.style_loss + self.total_variation_loss


    def _get_initial_image_or_random(self, initial):
        if initial is None:
            initial_image = tf.random_normal(self.content.shape)
        else:
            initial_image = self.vgg.preprocess(initial)
        return tf.Variable(initial_image)

    def _current_loss(self):
        losses = {}
        losses['content'] = self.content_loss.eval()
        losses['style'] = self.style_loss.eval()
        losses['total_variation'] = self.total_variation_loss.eval()
        losses['total'] = self.loss.eval()
        return losses

    def train(self, learning_rate, iterations, checkpoint_iterations):

        def is_last(i):
            return i == iterations - 1

        def is_checkpoint_iteration(i):
            return (checkpoint_iterations and i % checkpoint_iterations == 0) or is_last(i)

        def print_progress(i):
            stdout.write('Iteration %d/%d\n' % (i + 1, iterations))

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        best_loss = float('inf')
        best = None

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(iterations):
                print_progress(i)

                train_step.run()

                if is_checkpoint_iteration(i):
                    current_loss = self.loss.eval()
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best = self.image.eval()
                    yield (
                        (None if is_last(i) else i),
                        self.vgg.unprocess(best.reshape(self.content.shape[1:])),
                        self._current_loss()
                    )


class LossCalculator:

    def __init__(self, vgg, stylized_image):
        self.vgg = vgg
        self.network = vgg.net(stylized_image)

    def content_loss(self, content, content_layer, content_weight):
        # compute content features in feedforward mode
        content_image = tf.placeholder('float', shape=content.shape)
        content_net = self.vgg.net(content_image)

        with tf.Session() as sess:
            content_feature = content_net[content_layer].eval(
                    feed_dict={content_image: self.vgg.preprocess(content)})


        # content loss
        content_loss = content_weight * (2 * tf.nn.l2_loss(
                self.network[content_layer] - content_feature) /
                content_feature.size)

        return content_loss


    def style_loss(self, style, style_layers, style_weight):
        image = tf.placeholder('float', shape=style.shape)
        style_net = self.vgg.net(image)

        with tf.Session() as sess:
            style_preprocessed = self.vgg.preprocess(style)

            style_loss = 0

            for layer in style_layers:
                style_image_gram = self._calculate_style_gram_matrix_for(style_net,
                                                                   image,
                                                                   layer,
                                                                   style_preprocessed)

                input_image_gram = self._calculate_input_gram_matrix_for(layer)

                style_loss += style_weight * (2 * tf.nn.l2_loss(input_image_gram - style_image_gram) / style_image_gram.size)

        return style_loss

    def tv_loss(self, image, shape, tv_weight):
        # total variation denoising
        tv_y_size = _tensor_size(image[:,1:,:,:])
        tv_x_size = _tensor_size(image[:,:,1:,:])
        tv_loss = tv_weight * 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                    tv_x_size))

        return tv_loss

    def _calculate_style_gram_matrix_for(self, network, image, layer, style_image):
        image_feature = network[layer].eval(feed_dict={image: style_image})
        image_feature = np.reshape(image_feature, (-1, image_feature.shape[3]))
        return np.matmul(image_feature.T, image_feature) / image_feature.size

    def _calculate_input_gram_matrix_for(self, layer):
        image_feature = self.network[layer]
        _, height, width, number = map(lambda i: i.value, image_feature.get_shape())
        size = height * width * number
        image_feature = tf.reshape(image_feature, (-1, number))
        return tf.matmul(tf.transpose(image_feature), image_feature) / size

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)
