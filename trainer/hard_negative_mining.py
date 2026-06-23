"""Hard Negative Mining - 找出并保存误判图片。

从混淆矩阵中找出误判的图片，保存到 hard_negative 目录。
重点找出：
- screen_photo → screenshot 的误判（看起来像截图的拍屏）
- screenshot → screen_photo 的误判（看起来像拍屏的截图）
- natural → screenshot/screen_photo 的误判
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from . import config
from .augment import get_val_transforms
from .dataset import TwoInputDataset, compute_fft_spectrum
from .model import load_model


def find_misclassified_images(
    model: nn.Module,
    data_dir: Path | None = None,
    device: str = "cpu",
) -> dict[str, list[dict]]:
    """找出所有误判的图片。

    Args:
        model: 训练好的模型
        data_dir: 数据目录
        device: 设备

    Returns:
        字典，key 为误判类型，value 为误判图片列表
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    model.eval()

    # 创建数据集
    full_dataset = TwoInputDataset(
        data_map=config.THREE_CLASS_DATA_MAP,
        data_dir=data_dir,
        transform=None,
    )

    # 获取验证集索引
    total_size = len(full_dataset)
    train_size = int(total_size * config.TRAIN_VAL_SPLIT)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    indices = torch.randperm(total_size, generator=generator).tolist()
    val_indices = indices[train_size:]

    class_names = config.CLASS_NAMES_THREE_CLASS
    misclassified = {
        "screen_photo_to_screenshot": [],  # 最重要的误判
        "screenshot_to_screen_photo": [],
        "natural_to_screenshot": [],
        "natural_to_screen_photo": [],
        "screenshot_to_natural": [],
        "screen_photo_to_natural": [],
    }

    transform = get_val_transforms()

    print(f"验证集大小: {val_size} 张图片")
    print("开始分析...")

    for i, idx in enumerate(val_indices):
        img_path, true_label = full_dataset.samples[idx]

        # 加载和预处理图片
        image = np.array(Image.open(img_path).convert("RGBA").convert("RGB"))

        # RGB 分支
        if transform:
            augmented = transform(image=image)
            rgb_tensor = augmented["image"].unsqueeze(0).to(device)
        else:
            image_resized = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
            rgb_tensor = (
                torch.from_numpy(image_resized).permute(2, 0, 1).float().unsqueeze(0)
                / 255.0
            )
            rgb_tensor = rgb_tensor.to(device)

        # FFT 分支
        fft_spectrum = compute_fft_spectrum(image, config.IMAGE_SIZE)
        fft_tensor = torch.from_numpy(fft_spectrum).float().unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            outputs = model(rgb_tensor, fft_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        pred_label = int(predicted.item())
        true_class = class_names[true_label]
        pred_class = class_names[pred_label]

        if true_label != pred_label:
            misclassified[f"{true_class}_to_{pred_class}"].append(
                {
                    "path": Path(img_path),
                    "true_class": true_class,
                    "pred_class": pred_class,
                    "confidence": probs[0][pred_label].item(),
                }
            )

        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{val_size} 张图片")

    return misclassified


def copy_misclassified_to_hard_negative(
    misclassified: dict[str, list[dict]],
    output_dir: Path | None = None,
) -> dict[str, int]:
    """将误判图片复制到 hard_negative 目录。

    Args:
        misclassified: 误判图片字典
        output_dir: 输出目录

    Returns:
        各类误判的数量
    """
    if output_dir is None:
        output_dir = config.DATA_DIR / "hard_negative"

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for mis_type, items in misclassified.items():
        if not items:
            stats[mis_type] = 0
            continue

        # 创建子目录
        sub_dir = output_dir / mis_type
        sub_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for item in items:
            src_path = item["path"]
            if src_path.exists():
                # 使用原文件名，添加置信度后缀
                confidence = item["confidence"]
                dst_name = f"{src_path.stem}_conf{confidence:.2f}{src_path.suffix}"
                dst_path = sub_dir / dst_name

                shutil.copy2(src_path, dst_path)
                count += 1
                print(
                    f"  复制: {src_path.name} → {mis_type}/ (置信度: {confidence:.4f})"
                )

        stats[mis_type] = count

    return stats


def main():
    """主函数：执行 Hard Negative Mining。"""
    print("=" * 60)
    print("Hard Negative Mining - 误判图片挖掘")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 加载模型
    checkpoint_path = config.CHECKPOINT_DIR / "three_class_best.pth"
    if not checkpoint_path.exists():
        print(f"错误: 模型文件不存在 {checkpoint_path}")
        return None

    print(f"加载模型: {checkpoint_path}")
    model = load_model(str(checkpoint_path), device=device)
    model = model.to(device)

    # 找出误判图片
    print("\n分析验证集中的误判图片...")
    misclassified = find_misclassified_images(model, device=device)

    # 统计误判数量
    print("\n" + "=" * 60)
    print("误判统计")
    print("=" * 60)

    total_misclassified = 0
    for mis_type, items in misclassified.items():
        count = len(items)
        total_misclassified += count
        if count > 0:
            print(f"  {mis_type}: {count} 张")

    print(f"\n总计误判: {total_misclassified} 张")

    # 复制误判图片到 hard_negative 目录
    if total_misclassified > 0:
        print("\n" + "=" * 60)
        print("复制误判图片到 hard_negative 目录")
        print("=" * 60)

        output_dir = config.DATA_DIR / "hard_negative"
        stats = copy_misclassified_to_hard_negative(misclassified, output_dir)

        print("\n复制完成!")
        for mis_type, count in stats.items():
            if count > 0:
                print(f"  {mis_type}: {count} 张 → {output_dir / mis_type}/")

    # 打印重点误判详情
    print("\n" + "=" * 60)
    print("重点误判: screen_photo → screenshot")
    print("=" * 60)

    sp_to_ss = misclassified.get("screen_photo_to_screenshot", [])
    if sp_to_ss:
        print(f"共 {len(sp_to_ss)} 张图片被误判:")
        for i, item in enumerate(sp_to_ss[:10]):  # 只显示前10个
            print(f"  {i + 1}. {item['path'].name} (置信度: {item['confidence']:.4f})")
    else:
        print("没有此类误判!")

    return misclassified


if __name__ == "__main__":
    main()
