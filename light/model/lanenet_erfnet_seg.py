# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


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


class ERFNet(nn.Module):
    def __init__(self, num_classes, encoder=None):  # use encoder to pass pretrained encoder
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)  # predict=False by default
            return self.decoder.forward(output)


class LaneNetErfNetSeg(nn.Module):
    def __init__(
            self,
            embed_dim=4,
            delta_v=0.5,
            delta_d=3.0,
            scale_lane_line=1.0,
            scale_var=1.0,
            scale_dist=1.0,
            pretrained=False,
            **kwargs
    ):
        super(LaneNetErfNetSeg, self).__init__()
        self.pretrained = pretrained
        self.embed_dim = embed_dim
        self.delta_v = delta_v
        self.delta_d = delta_d

        self.net_init()

        self.scale_seg = scale_lane_line
        self.scale_var = scale_var
        self.scale_dist = scale_dist
        self.scale_reg = 0
        self.seg_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.4, 1.]))

    def net_init(self):
        self.backbone = Encoder(1)

        # ----------------- additional conv -----------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.PixelShuffle()
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )

        # ----------------- embedding -----------------
        self.embedding = nn.Sequential(
            nn.Conv2d(8, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, self.embed_dim, 1)
        )

        # ----------------- binary segmentation -----------------
        self.binary_seg = nn.Sequential(
            nn.Conv2d(8, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 2, 1)
        )

    def forward(self, img, segLabel=None):
        x = self.backbone(img)
        x = self.layer1(x)

        embedding = self.embedding(x)
        binary_seg = self.binary_seg(x)

        if segLabel is not None:
            var_loss, dist_loss, reg_loss = self.discriminative_loss(
                embedding, segLabel)
            seg_loss = self.seg_loss(
                binary_seg, torch.gt(segLabel, 0).type(torch.long))
        else:
            var_loss = torch.tensor(0, dtype=img.dtype, device=img.device)
            dist_loss = torch.tensor(0, dtype=img.dtype, device=img.device)
            seg_loss = torch.tensor(0, dtype=img.dtype, device=img.device)
            reg_loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        loss = seg_loss * self.scale_seg + var_loss * \
            self.scale_var + dist_loss * self.scale_dist

        output = {
            "embedding": embedding,
            "binary_seg": binary_seg,
            "loss_seg": seg_loss,
            "loss_var": var_loss,
            "loss_dist": dist_loss,
            "reg_loss": reg_loss,
            "loss": loss
        }

        return output

    def discriminative_loss(self, embedding, seg_gt):
        batch_size = embedding.shape[0]

        var_loss = torch.tensor(0, dtype=embedding.dtype,
                                device=embedding.device)
        dist_loss = torch.tensor(
            0, dtype=embedding.dtype, device=embedding.device)
        reg_loss = torch.tensor(0, dtype=embedding.dtype,
                                device=embedding.device)

        for b in range(batch_size):
            embedding_b = embedding[b]  # (embed_dim, H, W)
            seg_gt_b = seg_gt[b]

            labels = torch.unique(seg_gt_b)
            labels = labels[labels != 0]
            num_lanes = len(labels)
            if num_lanes == 0:
                # please refer to issue here: https://github.com/harryhan618/LaneNet/issues/12
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for lane_idx in labels:
                seg_mask_i = (seg_gt_b == lane_idx)
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[:, seg_mask_i]

                mean_i = torch.mean(embedding_i, dim=1)
                centroid_mean.append(mean_i)

                # ---------- var_loss -------------
                var_loss = var_loss + torch.mean(F.relu(torch.norm(embedding_i-mean_i.reshape(
                    self.embed_dim, 1), dim=0) - self.delta_v)**2) / num_lanes
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, self.embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, self.embed_dim)
                # shape (num_lanes, num_lanes)
                dist = torch.norm(centroid_mean1-centroid_mean2, dim=2)
                # diagonal elements are 0, now mask above delta_d
                dist = dist + \
                    torch.eye(num_lanes, dtype=dist.dtype,
                              device=dist.device) * self.delta_d

                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + \
                    torch.sum(F.relu(-dist + self.delta_d)**2) / \
                    (num_lanes * (num_lanes-1)) / 2

            # reg_loss is not used in original paper
            # reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / batch_size
        dist_loss = dist_loss / batch_size
        reg_loss = reg_loss / batch_size
        return var_loss, dist_loss, reg_loss


def get_lanenet_erfnet_seg(dataset='citys', pretrained=False, root='~/.torch/models',
                           pretrained_base=False, **kwargs):
    from light.data import datasets
    model = LaneNetErfNetSeg()
    if pretrained:
        from ..model import get_model_file
        model.load_state_dict(
            torch.load(get_model_file('erfnet_%s_best_model' % dataset, root=root), map_location='cpu'))
    return model


if __name__ == '__main__':
    # model = get_lanenet_erfnet_seg()
    model = LaneNetErfNetSeg()
    batchInputs = torch.randn(
        4, 3, 128, 256, dtype=torch.float, requires_grad=False)
    segLabel = torch.zeros(
        4, 128, 256, dtype=torch.float, requires_grad=False)
    b = model(batchInputs, segLabel)
    # make_dot(b).render("attached", format="png")
    print(b)
