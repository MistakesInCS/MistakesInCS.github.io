---

title: Failing to Learn to Decode
date: 2025-12-14 -0500
categories: [Project, Machine Learning]
tags: [optimization, machine learning, neural networks, project]
description: Trying to teach a neural network to learn to undo a compression algorithm, and then dumbing down the problem until I don't care any more.
math: true
mermaid: true

---
## Introduction
I would describe what I did for the last month as clumsy stumbling. Or stupidity. Probably some combination of the two. I knew this wouldn't work, and yet I'm still somehow disappointed. 

I saw a paper a while ago that said something like neural nets can't unencrypt data. As expected, to some degree. I don't know a thing about information theory, and I know even less about how it relates to cryptography, so please excuse my butchering of terminology. In my vague intuition, encrypted data is prepared in such a way that each bit of the ciphertext has very little information about any given bit in the cleartext.

So I thought, Ok, makes sense, obfuscated information hinders learning. What about compression? It's kinda like encryption in the sense that it transforms data chaotically from one form to another, except it's also like the opposite of encryption since the compressor explicitly wants you to be able to recover the data. I knew this was probably a fool's errand, but on the offchance I could get it to work it would be pretty good, and it's a good excuse to finally actually learn pytorch. I used MLPs, deconvolutions, LSTMs, and transformers, but spoiler, none of them really worked.

## First Attempts

To quote the [Fashion-MNIST readme,](https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md) "if it doesn't work on MNIST, it won't work at all." Therefore, I decided to work on MNIST data, and the plan is pretty simple: take the original MNIST image vector, compress it using zlib, then train neural nets using the compressed data as the example and image as the target. Getting pytorch to work with this was actually pretty simple and straightforward. To actually make the data, I used the following code:

```python
self.compressed_data = [(list(zlib.compress(str(torch.flatten(feature).tolist()).encode())), feature)
                        for feature in data]
```

First, I take the MNIST image data (feature) and flatten it to form a 1D tensor. Then, I convert that to a python list. Then, I encode it, which turns the string into its byte-level form. Then, I compress that with zlib, creating a compressed form of the original data. Casting that to a list then gives a list of integers, where the i-th integer represents the i-th byte in the sequence. Here's some interesting trivia: the largest compressed MNIST image is 1359 bytes, and the smallest is just 67. For uncompressed MNIST, the largest is 9044, and the smallest is 3920. That's a massive size improvement! 

I think this is pretty well tokenized. I used -1e9 for padding, but there are probably more sophisticated ways to do that for simple nnets. I use proper padding later when I pull out a transformer.

I first fed the data through several layers of MLPs, then followed up with a series of transposed convolutions (deconvolutions) to turn the 1D tensor into 2D image data. For this first run, I had the MLPs output a 512-long tensor, which then is directly fed into an initial 1x1 deconvolution with 1 channel per element in the tensor.

<details><summary><strong>Boring pytorch code implementing the model:</strong></summary>
```python
class NeuralNetwork(nn.Module):
  def __init__(self, max_size):
    super().__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
        nn.Linear(max_size, 2048),
        nn.LazyBatchNorm1d(),
        nn.LeakyReLU(),
        nn.Linear(2048, 2048),
        nn.LazyBatchNorm1d(),
        nn.LeakyReLU(),
        nn.Linear(2048, 2048),
        nn.LazyBatchNorm1d(),
        nn.LeakyReLU(),
        nn.Dropout(0.1),
        nn.Linear(2048, 512),
        nn.LazyBatchNorm1d(),
        #nn.LeakyReLU(),
    )

  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits

# Code lovingly lifted from https://d2l.ai/chapter_generative-adversarial-networks/dcgan.html
# Bias to false because it's followed by a batch norm and the subtraction cancels the bias - thanks google
class LatentTo2dBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False):
    super().__init__()
    self.TransposedConv2d = nn.ConvTranspose2d(in_channels, out_channels,
                                kernel_size, stride, padding, bias=bias)
    self.batch_norm = nn.BatchNorm2d(out_channels)
    self.activation = nn.LeakyReLU()
  def forward(self, x):
    return self.activation(self.batch_norm(self.TransposedConv2d(x)))

class LatentTo2d(nn.Module):
  def __init__(self):
    super().__init__()
    self.Deconvolve = nn.Sequential(
        # Sizes from https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        LatentTo2dBlock(in_channels=512, out_channels=128), # in: (512, 1, 1), out: (128, 2, 2)
        LatentTo2dBlock(in_channels=128, out_channels=32), # out: (32, 4, 4)
        LatentTo2dBlock(in_channels=32, out_channels=8, kernel_size=3, padding=(1,1)), # out: (8, 7, 7)
        LatentTo2dBlock(in_channels=8, out_channels=2), # out: (2, 14, 14)
        nn.ConvTranspose2d(in_channels=2, out_channels=1,
                           kernel_size=2, stride=2, padding=0, bias=True), # out: (1, 28, 28)
        nn.Sigmoid()
    )

  def forward(self, x):
    # Unsqueeze to go from size (512) to (512, 1, 1)
    return self.Deconvolve(x.unsqueeze(-1).unsqueeze(-1))

simple_nnet = nn.Sequential(NeuralNetwork(max_len), LatentTo2d())
```
</details>

Note that this is the final code that I ended up working on and is not the original model. I tweaked a lot of things to get it to that point, like changing ReLU to leaky ReLU, adding batch norms, etc.

Now I could actually start. As a first run, results were not looking good:

![An original MNIST 5](/assets/encoding/first5.png)
![A recreation of that MNIST 5 from the compressed version](/assets/encoding/first5r.png)
_An original MNIST 5 and its recreation from the compressed data_

At this point, I added batch norm layers and changed the optimizer to Adam. Slightly better, but....

![The new recreated 5](/assets/encoding/Blob1.png)
_The new recreated 5_

This definitely isn't a 5. In fact, it looks more like an 8, and there's definitely some highlighting around where a 3 would go. It's like one of those digital clocks with a segmented 8 that turns on and off each segment to make the different digits. One of my fears going into this was that the networks would just learn to make an "average" number instead of actually learn any kind of discerning information from the compression. 

I decided to let it train for a bit and got some cleaner looking numbers.

![An original 2 from the training set](/assets/encoding/comp1.png)
_An original 2 from the training set_

![A recreated 2 from the training set](/assets/encoding/comp1r.png){: .normal }
![A recreated 2 from the training set](/assets/encoding/comp1r2.png){: .normal }
![A recreated 2 from the training set](/assets/encoding/comp1r3.png){: .normal }
_Recreations of that same 2, but from different training epochs, increasing from left to right_

These 2s don't really look like the original very much; the first and third are clearly different styles of drawing a 2 - with a straight and angular bottom corner, instead of a loop. 

I thought that this might be due to overfitting, so I decided to take a look at the validation set. 

![An original 1 from the test set](/assets/encoding/test1.png){: .normal }
![A recreated 1 from the test set](/assets/encoding/test1r.png){: .normal }

![An original 0 from the test set](/assets/encoding/test2.png){: .normal }
![A recreated 0 from the test set](/assets/encoding/test2r.png){: .normal }

![An original 9 from the test set](/assets/encoding/test3.png){: .normal }
![A recreated 9 from the test set](/assets/encoding/test3r.png){: .normal }

![An original 5 from the test set](/assets/encoding/test4.png){: .normal }
![A recreated 5 from the test set](/assets/encoding/test4r.png){: .normal }

I had hoped that it would perform some classification task under the hood and then just generate a "good enough" candidate number for that class, but looking at the test set, it seems like it just generates a random number. Which is still kinda cool! It's neat to make a "random MNIST generator" with nothing but an MLP and deconvolutions.

Here's the loss curve, showing clear overfitting:

![Loss curve with overfitting](/assets/encoding/graph1.png)

I then switched to using MSELoss instead of the default, but overfitting still happened at around the same epoch and the numbers got blobbier.

![Loss curve with MSELoss](/assets/encoding/graph2.png)

![Original 2 from the training set](/assets/encoding/msetrain.png){: .normal }
![Rereated 2 from the training set](/assets/encoding/msetrainr.png){: .normal }
_From the training set_

![Original 4 from the test set](/assets/encoding/msetrain.png){: .normal }
![Rereated 4 from the test set](/assets/encoding/msetrainr.png){: .normal }
_From the test set_