from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import random
from random import randint
import lightning.pytorch as pl
from torch.utils.data import random_split
import torch
from torchvision.models import resnet50, ResNet50_Weights


random.seed(42)
weights = ResNet50_Weights.IMAGENET1K_V1
pretrained_viz_model = resnet50(weights=weights)
preprocess = weights.transforms()


class FlickrDataset(Dataset):
    def __init__(self, X, y, pipeline_img=None):
        super().__init__()
        self.X = X
        self.y = y
        self.pipeline_img = pipeline_img

        self.random_caption = False

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img_path = self.X[idx]
        caption_rand_idx = 1

        if self.random_caption:
            caption_rand_idx = randint(0, 4)

        input_id = self.y["input_ids"][idx][caption_rand_idx]
        attention_mask = self.y["attention_mask"][idx][caption_rand_idx]

        img = read_image(img_path)

        if self.pipeline_img:
            img = self.pipeline_img(img)

        return img, input_id, attention_mask


class FlickrDataModule(pl.LightningDataModule):
    def __init__(
        self,
        paths,
        tokenizer_output,
        batch_size,
        num_workers,
        random_caption,
        ratios=[6091, 1000, 1000],
    ):
        super().__init__()

        self.paths = paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer_output = tokenizer_output
        self.random_caption = random_caption
        self.ratios = ratios
        self.setup("pretrain")

    def setup(self, stage: str):
        dataset = FlickrDataset(self.paths, self.tokenizer_output, preprocess)

        generator_rand = torch.Generator().manual_seed(42)

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, self.ratios, generator=generator_rand
        )

        self.train_dataset.random_caption = self.random_caption

    def get_train_size(self):
        return len(self.train_dataset)

    def get_val_size(self):
        return len(self.val_dataset)

    def get_test_size(self):
        return len(self.test_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )


# data_mod = FlickrDataModule(df["img_path"], tokenizer_output, 128, 2, True)
