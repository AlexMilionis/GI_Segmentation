import os

import albumentations as A
import cv2
import lightning as L
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, random_split
import json


class KvasirSEGDatagen(Dataset):
    def __init__(self, pairs, transform=None):
        self.transform = transform
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image = cv2.imread(self.pairs[idx][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.pairs[idx][1], 0)
        mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1]

        print(type(image), type(mask))

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image_shape = image.shape
        # After ToTensorV2, image has shape [C, H, W], so check H == W (indices 1 and 2)
        if image_shape[1] != image_shape[2]:
            raise ValueError(f"Image shape mismatch: {image_shape}. Expected square images.")
        
        mask_shape = mask.shape
        if mask_shape[0] != mask_shape[1]:
            raise ValueError(f"Mask shape mismatch: {mask_shape}. Expected square masks.")
        
        print(f"Image path: {self.pairs[idx][0]}")
        print(f"Mask path: {self.pairs[idx][1]}")
                
        return image, mask.long().unsqueeze(0)


class KvasirSEGDataset(L.LightningDataModule):
    def __init__(
        self,
        batch_size=10,
        root_dir="./PolypGen",
        # root_dir="./Kvasir-SEG",
        num_workers=4,
        train_val_ratio=0.8,
        img_size=(224, 224),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.train_val_ratio = train_val_ratio
        self.img_size = img_size

    def get_train_transforms(self):
        return A.Compose(
            [
                A.Resize(*(self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(
                    p=0.5, brightness=(0.6, 1.6), contrast=0.2, saturation=0.1, hue=0.01
                ),
                A.Affine(
                    p=0.5,
                    scale=(0.5, 1.5),
                    translate_percent=0.125,
                    rotate=90,
                    interpolation=cv2.INTER_LANCZOS4,
                ),
                A.ElasticTransform(p=0.5, interpolation=cv2.INTER_LANCZOS4),
                A.Normalize(
                    # # ImageNet normalization (Kvasir-SEG)
                    # mean=(0.485, 0.456, 0.406),
                    # std=(0.229, 0.224, 0.225),
                    # Custom normalization for PolypGen
                    mean=(0.5543, 0.3644, 0.2777),
                    std=(0.2840, 0.2101, 0.1770),
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )

    def get_val_transforms(self):
        return A.Compose(
            [
                # A.Resize(*(self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4),
                A.Resize(*(self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4),
                A.Normalize(
                    # # ImageNet normalization (Kvasir-SEG)
                    # mean=(0.485, 0.456, 0.406),
                    # std=(0.229, 0.224, 0.225),
                    # Custom normalization for PolypGen
                    mean=(0.5543, 0.3644, 0.2777),
                    std=(0.2840, 0.2101, 0.1770),
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )

    def get_test_transforms(self):
        return A.Compose(
            [
                A.Resize(*(self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4),
                A.Normalize(
                    # # ImageNet normalization (Kvasir-SEG)
                    # mean=(0.485, 0.456, 0.406),
                    # std=(0.229, 0.224, 0.225),
                    # Custom normalization for PolypGen
                    mean=(0.5543, 0.3644, 0.2777),
                    std=(0.2840, 0.2101, 0.1770),
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )

    def setup(self, stage=None):
        if self.root_dir=="./PolypGen":
            train_images = os.listdir(os.path.join(self.root_dir, "train/images"))
            # train_masks = sorted(os.listdir(os.path.join(self.root_dir, "train/masks")))
            train_masks = []
            for img in train_images:
                base = img.split(".")[0]
                train_masks.append(base + "_mask.jpg")
            train_images = [os.path.join(self.root_dir, "train/images", img) for img in train_images]
            train_masks = [os.path.join(self.root_dir, "train/masks", mask) for mask in train_masks]

            val_images = sorted(os.listdir(os.path.join(self.root_dir, "validation/images")))
            # val_masks = sorted(os.listdir(os.path.join(self.root_dir, "validation/masks")))
            val_masks = []
            for img in val_images:
                base = img.split(".")[0]
                val_masks.append(base + "_mask.jpg")
            val_images = [os.path.join(self.root_dir, "validation/images", img) for img in val_images]
            val_masks = [os.path.join(self.root_dir, "validation/masks", mask) for mask in val_masks]

            test_images = sorted(os.listdir(os.path.join(self.root_dir, "test/images")))
            # test_masks = sorted(os.listdir(os.path.join(self.root_dir, "test/masks")))
            test_masks = []
            for img in test_images:
                base = img.split(".")[0]
                test_masks.append(base + "_mask.jpg")
            test_images = [os.path.join(self.root_dir, "test/images", img) for img in test_images]
            test_masks = [os.path.join(self.root_dir, "test/masks", mask) for mask in test_masks]

            pred_images = sorted(os.listdir(os.path.join(self.root_dir, "prediction/images")))
            pred_masks = []
            for img in pred_images:
                base = img.split(".")[0]
                pred_masks.append(base + "_mask.jpg")
            pred_images = [os.path.join(self.root_dir, "prediction/images", img) for img in pred_images]
            pred_masks = [os.path.join(self.root_dir, "prediction/masks", mask) for mask in pred_masks]

            train_pairs = list(zip(train_images, train_masks))
            # with open(os.path.join(self.root_dir, "train_pairs.json"), "w") as f:
            #     json.dump(train_pairs, f, indent=2)
            val_pairs = list(zip(val_images, val_masks))
            # with open(os.path.join(self.root_dir, "val_pairs.json"), "w") as f:
            #     json.dump(val_pairs, f, indent=2)
            test_pairs = list(zip(test_images, test_masks))
            # with open(os.path.join(self.root_dir, "test_pairs.json"), "w") as f:
            #     json.dump(test_pairs, f, indent=2)
            pred_pairs = list(zip(pred_images, pred_masks))

            # for img, mask in train_pairs:
            #     if img.split("\\")[-1].split('.')[0] + "_mask" != mask.split("\\")[-1].split('.')[0]:
            #         raise ValueError(f"Image and mask names do not match: {img} vs {mask}")
                
            # for img, mask in val_pairs:
            #     if img.split("\\")[-1].split('.')[0] + "_mask" != mask.split("\\")[-1].split('.')[0]:
            #         raise ValueError(f"Image and mask names do not match: {img} vs {mask}")
                
            # for img, mask in test_pairs:
            #     if img.split("\\")[-1].split('.')[0] + "_mask" != mask.split("\\")[-1].split('.')[0]:
            #         raise ValueError(f"Image and mask names do not match: {img} vs {mask}")
            
            # for img, mask in pred_pairs:
            #     if img.split("\\")[-1].split('.')[0] + "_mask" != mask.split("\\")[-1].split('.')[0]:
            #         raise ValueError(f"Image and mask names do not match: {img} vs {mask}") 
                
        elif self.root_dir=="./Kvasir-SEG":
            train_images = os.listdir(os.path.join(self.root_dir, "train/images"))
            train_masks = os.listdir(os.path.join(self.root_dir, "train/masks"))
            train_images = [os.path.join(self.root_dir, "train/images", img) for img in train_images]
            train_masks = [os.path.join(self.root_dir, "train/masks", mask) for mask in train_masks]

            val_images = os.listdir(os.path.join(self.root_dir, "validation/images"))
            val_masks = os.listdir(os.path.join(self.root_dir, "validation/masks"))
            val_images = [os.path.join(self.root_dir, "validation/images", img) for img in val_images]
            val_masks = [os.path.join(self.root_dir, "validation/masks", mask) for mask in val_masks]

            test_images = os.listdir(os.path.join(self.root_dir, "test/images"))
            test_masks = os.listdir(os.path.join(self.root_dir, "test/masks"))
            test_images = [os.path.join(self.root_dir, "test/images", img) for img in test_images]
            test_masks = [os.path.join(self.root_dir, "test/masks", mask) for mask in test_masks]

            pred_images = os.listdir(os.path.join(self.root_dir, "prediction/images"))
            pred_masks = os.listdir(os.path.join(self.root_dir, "prediction/masks"))
            pred_images = [os.path.join(self.root_dir, "prediction/images", img) for img in pred_images]
            pred_masks = [os.path.join(self.root_dir, "prediction/masks", mask) for mask in pred_masks]

            train_pairs = list(zip(train_images, train_masks))
            val_pairs = list(zip(val_images, val_masks))
            test_pairs = list(zip(test_images, test_masks))
            pred_pairs = list(zip(pred_images, pred_masks))

        self.train_set = KvasirSEGDatagen(train_pairs, transform=self.get_train_transforms())
        self.val_set = KvasirSEGDatagen(val_pairs, transform=self.get_val_transforms())
        self.test_set = KvasirSEGDatagen(test_pairs, transform=self.get_test_transforms())
        self.pred_set = KvasirSEGDatagen(pred_pairs, transform=self.get_test_transforms())

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.pred_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

