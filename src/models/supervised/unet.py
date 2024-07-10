import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn.functional import relu, pad


class DoubleConvHelper(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None):
        """
        Module that implements
            - a convolution
            - a batch norm
            - relu
            - another convolution
            - another batch norm
        """
        super(DoubleConvHelper, self).__init__()
        # if no mid_channels are specified, set mid_channels as out_channels
        if not mid_channels:
            mid_channels = out_channels
        # create a convolution from in_channels to mid_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size = 3,
                               padding = 1)  # kernel_size = 3, padding = 1, bias=false?
        # create a batch_norm2d of size mid_channels
        self.bn1 = nn.BatchNorm2d(mid_channels)
        # create a relu
        self.relu1 = nn.ReLU(inplace = True)
        # create a convolution from mid_channels to out_channels
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size = 3,
                               padding = 1)  # kernel_size = 3, padding = 1, bias=false?
        # create a batch_norm2d of size out_channels
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """Forward pass through the layers of the helper block"""
        # conv1
        x = self.conv1(x)
        # batch_norm1
        x = self.bn1(x)
        # relu
        x = self.relu1(x)
        # conv2
        x = self.conv2(x)
        # batch_norm2
        x = self.bn2(x)
        # relu
        x = self.relu1(x)
        return x


class Encoder(nn.Module):
    """ Downscale using the maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        # create a maxpool2d of kernel_size 2 and padding = 0
        self.maxpool = nn.MaxPool2d(kernel_size = 2, padding = 0)
        # create a doubleconvhelper
        self.double_conv = DoubleConvHelper(in_channels, out_channels)

    def forward(self, x):
        # maxpool2d
        x = self.maxpool(x)
        # doubleconv
        x = self.double_conv(x)
        return x


# given
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        # create up convolution using convtranspose2d from in_channels to in_channels//2
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size = 2,
                                     stride = 2)  # first arg could also just be in_channels or in_channels // 2, the latter is copied demo code so we'll see
        # use a doubleconvhelper from in_channels to out_channels
        self.double_conv = DoubleConvHelper(in_channels, out_channels)

    def forward(self, x1, x2):
        # step 1 x1 is passed through the convtranspose2d
        x1 = self.up(x1)
        # step 2 The difference between x1 and x2 is calculated to account for differences in padding
        diffY = x2.size()[2] - x1.size()[
            2]  # Might be switched with diffY and diffX, just have to wait and see!!!!
        diffX = x2.size()[3] - x1.size()[3]
        # step 3 x1 is padded (or not padded) accordingly
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # step 4 & 5
        # x2 represents the skip connection
        # Concatenate x1 and x2 together with torch.cat
        x = torch.cat([x2, x1], dim = 1)  # dim = 1

        # step 6 Pass the concatenated tensor through a doubleconvhelper
        x = self.double_conv(x)
        # step 7 Return output
        return x


class OutConv(nn.Module):  # I'm unsure of where this fits into unet due to the commented outline
    """ OutConv is the replacement of the final layer to ensure
    that the dimensionality of the output matches the correct number of
    classes for the classification task.
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # create a convolution with in_channels = in_channels and out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_encoders: int = 2,
                 embedding_size: int = 64, scale_factor: int = 50, **kwargs):
        """
        Implements a unet, a network where the input is downscaled
        down to a lower resolution with a higher amount of channels,
        but the residual images between encoders are saved
        to be concatednated to later stages, creatin the
        nominal "U" shape.

        In order to do this, we will need n_encoders-1 encoders.
        The first layer will be a doubleconvhelper that
        projects the in_channels image to an embedding_size
        image of the same size.

        After that, n_encoders-1 encoders are used which halve
        the size of the image, but double the amount of channels
        available to them (i.e, the first layer is
        embedding_size -> 2*embedding size, the second layer is
        2*embedding_size -> 4*embedding_size, etc)

        The decoders then upscale the image and halve the amount of
        embedding layers, i.e., they go from 4*embedding_size->2*embedding_size.

        We then have a maxpool2d that scales down the output to by scale_factor,
        as the input for this architecture must be the same size as the output,
        but our input images are 800x800 and our output images are 16x16.
        """
        super(UNet, self).__init__()

        # save in_channels, out_channels, n_encoders, embedding_size to self
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_encoders = n_encoders
        self.embedding_size = embedding_size
        self.scale_factor = scale_factor
        # create a doubleconvhelper
        self.initial_conv = DoubleConvHelper(in_channels, embedding_size)

        # for each encoder (there's n_encoders encoders)
        encoders_temp = []
        current_channels = embedding_size
        for i in range(n_encoders):
            # append a new encoder with embedding_size as input and 2*embedding_size as output
            next_channels = 2 * current_channels
            encoders_temp.append(Encoder(current_channels, next_channels))
            # double the size of embedding_size
            current_channels = next_channels
        # store it in self.encoders as an nn.ModuleList
        self.encoders = nn.ModuleList(encoders_temp)

        # for each decoder (there's n_encoders decoders)
        decoders_temp = []
        for i in range(n_encoders):
            # if it's the last decoder
            if i == n_encoders - 1:
                # create a decoder of embedding_size input and out_channels output
                decoders_temp.append(Decoder(current_channels, out_channels))
                # create a decoder of embeding_size input and embedding_size//2 output

            else:  # Changed the implementation to create decoders normally like the encoder loop until the last layer, in which it maps to specified out channels
                decoders_temp.append(Decoder(current_channels, current_channels // 2))
            # halve the embedding size
            current_channels = current_channels // 2

        # save the decoder list as an nn.ModuleList to self.decoders
        self.decoders = nn.ModuleList(decoders_temp)

        # create a MaxPool2d of kernel size scale_factor as the final pooling layer
        self.final_pool = nn.MaxPool2d(kernel_size = scale_factor)

    def forward(self, x):
        """
            The image is passed through the encoder layers,
            making sure to save the residuals in a list.

            Following this, the residuals are passed to the
            decoder in reverse, excluding the last residual
            (as this is used as the input to the first decoder).

            The ith decoder should have an input of shape
            (batch, some_embedding_size, some_width, some_height)
            as the input image and
            (batch, some_embedding_size//2, 2*some_width, 2*some_height)
            as the residual.
        """
        # evaluate x with self.inc
        x = self.initial_conv(x)
        # create a list of the residuals, with its only element being x
        residuals = [x]
        # for each encoder
        for encoder in self.encoders:
            # run the residual through the encoder, append the output to the residual
            x = encoder(x)
            residuals.append(x)

        # set x to be the last value from the residuals
        x = residuals.pop()
        # for each residual except the last one
        for decoder, residual in zip(self.decoders, reversed(residuals)):
            # evaluate it with the decoder
            x = decoder(x, residual)

        # evaluate the final pooling layer
        x = self.final_pool(x)

        # return x
        return x
