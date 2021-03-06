# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """

    def __init__(self, input_channels, filter_size, num_features, device='cpu', isValMode=False, isNoMemory=False):
        super(CLSTM_cell, self).__init__()
        self.device = device
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.isValMode = isValMode
        self.isNoMemory = isNoMemory
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))
        self.hidden_state = None
        self.counter = 0

    def forward(self, inputs, hidden_state=None):
        # inputs [N,C,H,W]
        seq_len = inputs.shape[0]
        channel = inputs.shape[1]
        height = inputs.shape[2]
        width = inputs.shape[3]
        if self.hidden_state is None:
            hx = torch.zeros(1, channel, height,
                             width, device=self.device, requires_grad=False if self.isValMode else True)
            cx = torch.zeros(1, channel, height,
                             width, device=self.device, requires_grad=False if self.isValMode else True)
            self.hidden_state = (hx, cx)
        else:
            hx, cx = self.hidden_state
        output_inner = []
        for index in range(seq_len):
            x = inputs[index, ...].unsqueeze(0)  # [C,H,W] to [1,C,H,W]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)  # gates: S, num_features*4, H, W
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        if self.isValMode and (not self.isNoMemory):
            if self.counter > 5:
                self.counter = 0
                self.hidden_state = (hx, cx)
            else:
                self.counter = self.counter+1
        return torch.stack(output_inner).squeeze(1), (hy, cy)


class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput,
                              (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            1*dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 1*dilated), bias=True, dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output+input)  # +input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(
            128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(
            16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

# ERFNet


class ERFNetLstm(nn.Module):
    # use encoder to pass pretrained encoder
    def __init__(self,
                 num_classes,
                 encoder=None,
                 pretrainWeightFile=None,
                 device='cpu',
                 isValMode=False,
                 isNoMemory=False
                 ):
        super().__init__()
        self.device = device
        self.isValMode = isValMode
        self.isNoMemory = isNoMemory

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)
        self.clstm = CLSTM_cell(
            128, 3, 128, device=self.device, isValMode=self.isValMode, isNoMemory=self.isNoMemory)
        if pretrainWeightFile is not None:
            self.load_state_dict(
                torch.load(pretrainWeightFile, map_location='cpu'), strict=False)

    def forward(self, batchInputs, only_encode=False):
        if not self.isValMode:
            #inputs : [N,SEQ_LEN,C,H,W]
            batchOutputs = []
            for inputs in batchInputs:
                intermidiateOutputs = self.encoder(inputs)
                intermidiateOutputs, _ = self.clstm(intermidiateOutputs)
                outputs = self.decoder.forward(intermidiateOutputs)
                batchOutputs.append(outputs)
            return torch.stack(batchOutputs)
        else:
            #inputs : [N,C,H,W]
            intermidiateOutputs = self.encoder(batchInputs)
            intermidiateOutputs, _ = self.clstm(intermidiateOutputs)
            outputs = self.decoder.forward(intermidiateOutputs)
            return outputs


def get_erfnet_lstm_seg(dataset='citys', pretrained=False, root='~/.torch/models',
                        pretrained_base=False, **kwargs):
    from light.data import datasets
    model = ERFNetLstm(datasets[dataset].NUM_CLASS, **kwargs)
    if pretrained:
        from ..model import get_model_file
        model.load_state_dict(
            torch.load(get_model_file('erfnet_lstm_%s_best_model' % dataset, root=root), map_location='cpu'))
    return model


if __name__ == '__main__':
    # from torchviz import make_dot
    model = ERFNetLstm(1, isValMode=True)
    model.eval()
    # batchInputs = torch.randn(
    #     2, 4, 3, 128, 256, dtype=torch.float, requires_grad=False)
    batchInputs = torch.randn(
        1, 3, 128, 256, dtype=torch.float, requires_grad=False)
    b = model(batchInputs)
    # make_dot(b).render("attached", format="png")
    print(b.size())
