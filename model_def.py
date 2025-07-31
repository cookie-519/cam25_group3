import torch
import torch.nn as nn

# Generator Convolutional Block: This block can be used for both downsampling (encoder) and upsampling (decoder)
class GenConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_down=True, is_relu=True, use_dropout=True, is_norm=True):
        """
        Initialize the Generator Convolutional Block.

        Parameters:
        - in_channels: Number of input channels
        - out_channels: Number of output channels
        - is_down: If True, this block will be used for downsampling (conv), otherwise for upsampling (conv_transpose)
        - is_relu: If True, ReLU will be used as activation, otherwise LeakyReLU is used
        - use_dropout: If True, dropout will be applied after the activation
        - is_norm: If True, Instance Normalization will be applied after the convolution
        """
        super().__init__()

        layers = nn.ModuleList()  # List to hold layers for the block

        # Convolution layer (Down or Up)
        if is_down:  # Downsampling (Conv layer)
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,  # 4x4 kernel
                    stride=2,  # Stride of 2 for downsampling
                    padding=1,  # Padding to maintain spatial dimensions
                    padding_mode="reflect"  # Reflect padding to avoid border artifacts
                )
            )
        else:  # Upsampling (ConvTranspose layer)
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,  # 4x4 kernel for transposed convolution
                    stride=2,  # Stride of 2 for upsampling
                    padding=1  # Padding to maintain spatial dimensions
                )
            )

        # Instance Normalization (if required)
        if is_norm:
            layers.append(
                nn.InstanceNorm2d(num_features=out_channels)  # Normalize across each feature map
            )

        # Activation (ReLU or LeakyReLU)
        if is_relu:
            layers.append(
                nn.ReLU()  # Standard ReLU activation
            )
        else:
            layers.append(
                nn.LeakyReLU(0.2)  # LeakyReLU with negative slope of 0.2
            )

        # Dropout (if required)
        if use_dropout:
            layers.append(
                nn.Dropout(0.5)  # Apply dropout with a probability of 50%
            )

        # Sequential model containing all layers
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the block.

        Parameters:
        - x: Input tensor

        Returns:
        - out: Output tensor after applying all layers in the block
        """
        out = self.block(x)  # Pass input through the sequential layers
        return out



# Generator class for building the architecture of a deep learning model

class Generator(nn.Module):

    def __init__(self, in_channels= 3, num_features= 64):
        super().__init__()

        # DOWN LAYERS (Convolutional blocks for encoding the input image)

        # d1: (B, 3, 256, 256) -> (B, 64, 128, 128)
        self.down_layer_1 = GenConvBlock(
            in_channels=in_channels,
            out_channels=num_features,
            is_down=True,
            is_relu=False,
            use_dropout=False,
            is_norm=False
        )

        # d2: (B, 64, 128, 128) -> (B, 128, 64, 64)
        self.down_layer_2 = GenConvBlock(
            in_channels=num_features,
            out_channels=num_features*2,
            is_down=True,
            is_relu=False,
            use_dropout=False,
            is_norm=True
        )

        # d3: (B, 128, 64, 64) -> (B, 256, 32, 32)
        self.down_layer_3 = GenConvBlock(
            in_channels=num_features*2,
            out_channels=num_features*4,
            is_down=True,
            is_relu=False,
            use_dropout=False,
            is_norm=True
        )

        # d4: (B, 256, 32, 32) -> (B, 512, 16, 16)
        self.down_layer_4 = GenConvBlock(
            in_channels=num_features*4,
            out_channels=num_features*8,
            is_down=True,
            is_relu=False,
            use_dropout=False,
            is_norm=True
        )

        # d5: (B, 512, 16, 16) -> (B, 512, 8, 8)
        self.down_layer_5 = GenConvBlock(
            in_channels=num_features*8,
            out_channels=num_features*8,
            is_down=True,
            is_relu=False,
            use_dropout=False,
            is_norm=True
        )

        # d6: (B, 512, 8, 8) -> (B, 512, 4, 4)
        self.down_layer_6 = GenConvBlock(
            in_channels=num_features*8,
            out_channels=num_features*8,
            is_down=True,
            is_relu=False,
            use_dropout=False,
            is_norm=True
        )

        # d7: (B, 512, 4, 4) -> (B, 512, 2, 2)
        self.down_layer_7 = GenConvBlock(
            in_channels=num_features*8,
            out_channels=num_features*8,
            is_down=True,
            is_relu=False,
            use_dropout=False,
            is_norm=True
        )

        # BOTTLENECK (End of down-sampling)
        # d7: (B, 512, 2, 2) -> (B, 512, 1, 1)
        self.bottleneck = GenConvBlock(
            in_channels=num_features*8,
            out_channels=num_features*8,
            is_down=True,
            is_relu=True,
            use_dropout=False,
            is_norm=False
        )

        # UP LAYERS (Decoding or up-sampling the image)

        # u1: (B, 512, 1, 1) -> (B, 512, 2, 2)
        self.up_layer_1 = GenConvBlock(
            in_channels=num_features*8,
            out_channels=num_features*8,
            is_down=False,
            is_relu=True,
            use_dropout=True,
            is_norm=True
        )

        # u2: (B, 512, 2, 2) + (B, 512, 2, 2) = (B, 1024, 2, 2) -> (B, 512, 4, 4)
        self.up_layer_2 = GenConvBlock(
            in_channels=num_features*8*2,
            out_channels=num_features*8,
            is_down=False,
            is_relu=True,
            use_dropout=True,
            is_norm=True
        )

        # u3: (B, 512, 4, 4) + (B, 512, 4, 4) = (B, 1024, 4, 4) -> (B, 512, 8, 8)
        self.up_layer_3 = GenConvBlock(
            in_channels=num_features*8*2,
            out_channels=num_features*8,
            is_down=False,
            is_relu=True,
            use_dropout=True,
            is_norm=True
        )

        # u4: (B, 512, 8, 8) + (B, 512, 8, 8) = (B, 1024, 8, 8) -> (B, 512, 16, 16)
        self.up_layer_4 = GenConvBlock(
            in_channels=num_features*8*2,
            out_channels=num_features*8,
            is_down=False,
            is_relu=True,
            use_dropout=True,
            is_norm=True
        )

        # u5: (B, 512, 16, 16) + (B, 512, 16, 16) = (B, 1024, 16, 16) -> (B, 256, 32, 32)
        self.up_layer_5 = GenConvBlock(
            in_channels=num_features*8*2,
            out_channels=num_features*4,
            is_down=False,
            is_relu=True,
            use_dropout=True,
            is_norm=True
        )

        # u6: (B, 256, 32, 32) + (B, 256, 32, 32) = (B, 512, 32, 32) -> (B, 128, 64, 64)
        self.up_layer_6 = GenConvBlock(
            in_channels=num_features*4*2,
            out_channels=num_features*2,
            is_down=False,
            is_relu=True,
            use_dropout=True,
            is_norm=True
        )

        # u7: (B, 128, 64, 64) + (B, 128, 64, 64) = (B, 256, 64, 64) -> (B, 64, 128, 128)
        self.up_layer_7 = GenConvBlock(
            in_channels=num_features*2*2,
            out_channels=num_features,
            is_down=False,
            is_relu=True,
            use_dropout=True,
            is_norm=True
        )

        # Final output layer: (B, 64, 128, 128) + (B, 64, 128, 128) = (B, 128, 128, 128) -> (B, 3, 256, 256)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=num_features*2,
                out_channels=in_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()  # Use tanh activation to get pixel values between [-1, 1]
        )

    def forward(self, x):
        # Forward pass through the network

        # DOWN LAYERS
        d1 = self.down_layer_1(x)  # (B, 64, 128, 128)
        d2 = self.down_layer_2(d1)  # (B, 128, 64, 64)
        d3 = self.down_layer_3(d2)  # (B, 256, 32, 32)
        d4 = self.down_layer_4(d3)  # (B, 512, 16, 16)
        d5 = self.down_layer_5(d4)  # (B, 512, 8, 8)
        d6 = self.down_layer_6(d5)  # (B, 512, 4, 4)
        d7 = self.down_layer_7(d6)  # (B, 512, 2, 2)

        # BOTTLENECK
        bot_neck_out = self.bottleneck(d7)  # (B, 512, 1, 1)

        # UP LAYERS
        u1 = self.up_layer_1(bot_neck_out)  # (B, 512, 2, 2)
        u2 = self.up_layer_2(torch.cat([u1, d7], dim=1))  # (B, 512, 4, 4)
        u3 = self.up_layer_3(torch.cat([u2, d6], dim=1))  # (B, 512, 8, 8)
        u4 = self.up_layer_4(torch.cat([u3, d5], dim=1))  # (B, 512, 16, 16)
        u5 = self.up_layer_5(torch.cat([u4, d4], dim=1))  # (B, 256, 32, 32)
        u6 = self.up_layer_6(torch.cat([u5, d3], dim=1))  # (B, 128, 64, 64)
        u7 = self.up_layer_7(torch.cat([u6, d2], dim=1))  # (B, 64, 128, 128)

        final_out = self.final_layer(torch.cat([u7, d1], dim=1))  # (B, 3, 256, 256)

        return final_out
