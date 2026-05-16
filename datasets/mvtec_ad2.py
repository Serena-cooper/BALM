from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class MVTecAD2TestDataset(Dataset):
    IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(
        self,
        path,
        category,
        transform=T.Compose([T.ToTensor()]),
        split="test_private",
    ):
        super().__init__()

        self.root = Path(path).resolve()
        self.category = category
        self.split = split
        self.transform = transform

        self.category_root = self.root / category
        self.image_root = self.category_root / split

        if not self.category_root.exists():
            raise FileNotFoundError(f"Category folder not found: {self.category_root}")
        if not self.image_root.exists():
            raise FileNotFoundError(f"Image split folder not found: {self.image_root}")

        self.files = sorted(
            p
            for p in self.image_root.iterdir()
            if p.is_file() and p.suffix.lower() in self.IMG_EXTENSIONS
        )
        if len(self.files) == 0:
            raise RuntimeError(f"No test images found under {self.image_root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_path = self.files[index]

        image_pil = Image.open(image_path).convert("RGB")
        width, height = image_pil.size
        image = self.transform(image_pil)


        return {
            "image": image,
            "path": str(image_path.relative_to(self.root)).replace("\\", "/"),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
        }