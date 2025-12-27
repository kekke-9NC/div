

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import config


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        out += identity; out = self.relu(out)
        return out

class ComplexCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ComplexCNN, self).__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.classifier(x)
        return x

# =======================================================
# --- モデルと画像前処理の初期化 ---
# =======================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ComplexCNN(num_classes=2).to(device)
model_path = config.MODEL_PATH

try:
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        print("Warning: weights_only=True is not supported. Loading without it.")
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"モデルを {model_path} からロードしました。")
    model.eval()
except FileNotFoundError:
    print(f"エラー: モデルファイルが見つかりません: {model_path}")
    sys.exit(1)
except Exception as e:
    print(f"モデルのロード中に予期せぬエラーが発生しました: {e}")
    sys.exit(1)


DATASET_MEAN = [0.035, 0.035, 0.035]
DATASET_STD = [0.047, 0.047, 0.047]

print(f"Using custom normalization: MEAN={DATASET_MEAN}, STD={DATASET_STD}")

transform = transforms.Compose([
    transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
])


def predict_meteor_probability(image_path: str) -> float:
    """
    与えられた画像パスから画像を読み込み、モデルを使って流星である確率を予測する。
    TTA（Test-Time Augmentation）を適用し、予測の安定性と精度を向上させる。
    """
    try:
        pil_image = Image.open(image_path).convert("RGB")
        tta_transforms = [
            lambda x: x,
            lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
            lambda x: x.transpose(Image.FLIP_TOP_BOTTOM)
        ]
        all_probabilities = []
        with torch.no_grad():
            for tta_transform in tta_transforms:
                transformed_image = tta_transform(pil_image)
                image_tensor = transform(transformed_image)
                image_tensor = image_tensor.unsqueeze(0).to(device)
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                all_probabilities.append(probabilities)
            
            avg_probabilities = torch.stack(all_probabilities).mean(dim=0).squeeze(0)
            
            probability = avg_probabilities[0].item()
            
        return probability
    except FileNotFoundError:
        print(f"エラー: 推論用画像が見つかりません: {image_path}")
        return 0.0
    except Exception as e:
        print(f"推論中にエラーが発生しました ({image_path}): {e}")
        return 0.0


if __name__ == '__main__':
    print("model.py が直接実行されました。")
    print(f"モデルクラス: {type(model)}")
    print(f"モデルは評価モード: {not model.training}")
    print(f"使用デバイス: {device}")
    print("データ変換:")
    print(transform)

    try:
        import numpy as np
        import os
        dummy_image = Image.new('RGB', (config.IMG_WIDTH, config.IMG_HEIGHT), color = 'red')
        dummy_image_path = "dummy_test_image.png"
        dummy_image.save(dummy_image_path)
        prob = predict_meteor_probability(dummy_image_path)
        print(f"ダミー画像の流星確率 (TTA適用): {prob:.4f}")
        os.remove(dummy_image_path)
    except Exception as e:
        print(f"初期化テスト中にエラー: {e}")