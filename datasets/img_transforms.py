from torchvision import transforms

def default_transform(img_size=224):
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )


def swm_transform(img_size=448):
    """Pass-through for SWM-Next data; frames are pre-resized + kept in [0, 1]
    so the Qwen3-VL encoder can apply its CLIP-style normalization internally.
    """
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
        ]
    )