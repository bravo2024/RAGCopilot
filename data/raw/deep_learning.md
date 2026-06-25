# Deep Learning

## Neural networks

A neural network consists of layers of neurons, each computing a weighted sum of its inputs followed by a non-linear activation function like ReLU or sigmoid.

## CNNs

Convolutional neural networks use convolution kernels to extract spatial features from grid-like data such as images. Pooling layers reduce spatial dimensions progressively.

## RNNs

Recurrent neural networks maintain a hidden state that captures information from previous time steps, making them suitable for sequential data like text and time series.

## Transformers

Transformers replace recurrence with self-attention, allowing parallel processing of all positions in a sequence. They are the backbone of modern LLMs like GPT and BERT.

## Transfer learning

Transfer learning takes a model pre-trained on a large dataset and fine-tunes it on a smaller domain-specific dataset. It drastically reduces data and compute requirements.

## Batch normalisation

Batch normalisation normalises layer inputs across a mini-batch to stabilise training, allowing higher learning rates and reducing sensitivity to initialisation.

## Dropout

Dropout randomly sets a fraction of neurons to zero during training, forcing the network to learn redundant representations and reducing overfitting.

## Autoencoders

An autoencoder learns to compress input data into a latent representation and then reconstruct it. Variants include denoising autoencoders and variational autoencoders.

## GANs

Generative adversarial networks pit a generator against a discriminator. The generator produces synthetic data; the discriminator tries to tell real from fake.

## Attention mechanisms

Attention mechanisms let a model focus on relevant parts of the input when producing each output element. Self-attention computes attention over the input itself.
