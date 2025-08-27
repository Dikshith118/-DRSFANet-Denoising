import torch
import torch.nn as nn
import torch.fft

def create_dct_matrix(size):
    """Helper function to create a DCT matrix."""
    matrix = torch.zeros(size, size)
    pi = torch.tensor(torch.pi)
    for k in range(size):
        for n in range(size):
            if k == 0:
                matrix[k, n] = 1.0 / torch.sqrt(torch.tensor(size, dtype=torch.float32))
            else:
                angle = pi * (2 * n + 1) * k / (2.0 * size)
                matrix[k, n] = torch.sqrt(torch.tensor(2.0 / size)) * torch.cos(angle)
    return matrix

def dct_2d(x):
    """Performs a 2D DCT on a batch of square images."""
    B, C, H, W = x.shape
    if H != W:
        raise ValueError("Input for DCT must be square.")
    
    dct_matrix = create_dct_matrix(H).to(x.device)
    dct_res = dct_matrix @ x @ dct_matrix.T
    return dct_res

def idct_2d(x):
    """Performs a 2D Inverse DCT on a batch of square images."""
    B, C, H, W = x.shape
    if H != W:
        raise ValueError("Input for IDCT must be square.")
        
    dct_matrix = create_dct_matrix(H).to(x.device)
    idct_res = dct_matrix.T @ x @ dct_matrix
    return idct_res


class ResidualModule(nn.Module):
    def __init__(self, channels):
        super(ResidualModule, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x

class FeatureExtractionBlock(nn.Module):
    def __init__(self, in_channels):
        super(FeatureExtractionBlock, self).__init__()
        self.prelim_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.module1 = ResidualModule(64)
        self.trans1 = nn.Conv2d(64, 128, kernel_size=1)
        self.module2 = ResidualModule(128)
        self.trans2 = nn.Conv2d(128, 256, kernel_size=1)
        self.module3 = ResidualModule(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.prelim_conv(x))
        x = self.module1(x)
        x = self.relu(self.trans1(x))
        x = self.module2(x)
        x = self.relu(self.trans2(x))
        x = self.module3(x)
        return x

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels=256, out_channels=128):
        super(DownsamplingBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=64):
        super(ResidualAttentionBlock, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = self.relu(self.conv_in(x))
        x1 = self.relu(self.conv1(residual))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.sigmoid(self.conv4(x3))
        out = x1 * x4
        out = out + residual
        return out

class FrequencyPlaneAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(FrequencyPlaneAttentionBlock, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_dilated1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3)
        self.conv_dilated2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=4, dilation=4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)
        pool_combine = self.relu(self.conv_pool(max_out + avg_out))
        fpab1 = self.sigmoid(pool_combine)
        fpab2 = self.relu(self.conv_dilated1(x))
        fpab2 = self.relu(self.conv_dilated2(fpab2))
        out = fpab1 * fpab2
        return out

class DRSFANet(nn.Module):
    def __init__(self, in_channels):
        super(DRSFANet, self).__init__()
        self.feb = FeatureExtractionBlock(in_channels=in_channels)
        self.ds = DownsamplingBlock(in_channels=256, out_channels=128)
        self.rab = ResidualAttentionBlock(in_channels=128, out_channels=64)
        self.fpab = FrequencyPlaneAttentionBlock(in_channels=in_channels)
        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x):
        original_input = x.clone()
        spatial_features = self.feb(x)
        spatial_features = self.ds(spatial_features)
        spatial_features = self.rab(spatial_features)
        spatial_features = self.final_conv(spatial_features)
        
        freq_features = dct_2d(x)
        freq_features = self.fpab(freq_features)
        freq_features = idct_2d(freq_features)
        
        residual_noise = freq_features * spatial_features
        return residual_noise