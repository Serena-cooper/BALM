import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from transformers import AutoModel

from peft_local.peft_func import add_peft

from pathlib import Path
from unittest.mock import patch

import torch.nn as nn
from transformers import AutoModel
import huggingface_hub


class TIPSv2(nn.Module):

    def __init__(self, height):
        super().__init__()

        model_path = Path("./tipsv2-so400m14").resolve()
        original_hf_hub_download = huggingface_hub.hf_hub_download

        def hf_hub_download_local_first(repo_id, filename, *args, **kwargs):
            repo_path = Path(str(repo_id)).expanduser()

            if repo_path.exists():
                local_file = repo_path / filename
                if local_file.exists():
                    return str(local_file)

            local_file = model_path / filename
            if local_file.exists():
                return str(local_file)

            return original_hf_hub_download(repo_id, filename, *args, **kwargs)

        with patch("huggingface_hub.hf_hub_download", hf_hub_download_local_first):
            self.net = AutoModel.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                local_files_only=True,
            ).vision_encoder

        self.feature_dim = 1152
        self.patch_size = 14
        self.H = height

    def get_img_transform(self):
        return T.Compose(
            [
                T.Resize((self.H, self.H)),
                T.ToTensor(),
            ]
        )

    def add_peft(self, r=64, peft_type="lora"):
        add_peft(self.net, r=r, peft_type=peft_type)


    def forward(self, x):
        cls_1, cls_2, ftrs = self.net(x)
        summary = torch.cat([cls_1, cls_2], dim=2).squeeze(1)
        return summary, ftrs
