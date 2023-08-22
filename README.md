# ResNet-50_basic_struct

# ResNet (Residual Network)

**ResNet** is a deep learning architecture designed to facilitate the training of deep neural networks. The key innovation is the introduction of "residual connections" or "skip connections".

## Deep Dive: Residual Connections

Traditional deep neural networks can suffer from vanishing and exploding gradient problems as they grow deeper. ResNet's architecture addresses this by introducing "residual connections."

### Residual Block:

A typical ResNet block has two paths:
1. **Main Path:** This involves a series of convolutional, normalization, and activation layers.
2. **Shortcut (or Residual) Path:** This bypasses the main layers and connects the input directly to the output of the block.

When these paths are combined, the network essentially learns the "residual" or difference between the identity function and the desired transformation.

### The Identity Map:

The shortcut path in a ResNet block acts as an identity map, allowing the input to flow unchanged to the output. This simple idea ensures that the deep network's output is at least as good as its input, effectively ensuring that added layers don’t degrade performance. It simplifies the optimization landscape and makes gradients flow easier across layers.

The intuition is that it's easier to optimize the residual (or the difference) than to learn the entire transformation. This allows ResNets to be significantly deeper while being trainable and resistant to the issues associated with depth.

## Features at a Glance

- **Depth:** Variants exist, with ResNet50 having 50 layers.
- **Batch Normalization:** Normalizes layer activations.
- **Activation:** Uses ReLU post batch normalization.
- **Pooling:** Employs max and average pooling.
- **Parameters:** Efficient architecture; ResNet50 has ~25.6 million parameters.
