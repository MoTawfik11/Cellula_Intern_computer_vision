# models.py
import torch
import torch.nn as nn
import torchvision.models as models

# -------------------------
# 3D-CNN model (simple)
# -------------------------
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # expects input [B, C, T, H, W]
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2)),

            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2)),

            nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1,1,1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# -------------------------
# CNN + RNN model
# -------------------------
class CNN_RNN(nn.Module):
    def __init__(self, backbone_pretrained=True, rnn_hidden=256, num_classes=2):
        super().__init__()
        # Load ResNet18 pretrained
        try:
            self.backbone = models.resnet18(pretrained=backbone_pretrained)
        except TypeError:
            # some torchvision versions expect weights arg
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if backbone_pretrained else None)

        # remove fc layer
        modules = list(self.backbone.children())[:-1]  # remove last fc
        self.feature_extractor = nn.Sequential(*modules)  # outputs [B,512,1,1]
        self.rnn = nn.GRU(input_size=512, hidden_size=rnn_hidden, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(rnn_hidden, num_classes)

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        # move time dimension to batch for CNN processing
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # [B*T, C, H, W]
        feats = self.feature_extractor(x)  # [B*T, 512, 1, 1]
        feats = feats.view(B, T, -1)  # [B, T, 512]
        out, h = self.rnn(feats)      # out [B,T,hidden], h [1,B,hidden]
        last = h[-1]                  # [B, hidden]
        logits = self.classifier(last)
        return logits
