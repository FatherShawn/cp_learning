from typing import Tuple, List, Union, Callable
import torch as pt
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.nn import functional as F
from cp_image_dataset import QuackImageDataset
from cp_flatten import QuackConstants


class QuackImageTransformer:

    def __init__(self, step_type: str) -> None:
        super().__init__()
        valid_types = {'train', 'val', 'predict'}
        if step_type not in valid_types:
            raise ValueError('Improper transform step_type.')
        self.__type = step_type
        if self.__type == 'train':
            self.__transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=224, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomHorizontalFlip(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        if self.__type == 'val' or self.__type == 'predict':
            self.__transforms = transforms.Compose([
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __call__(self, batch: List[dict], *args, **kwargs) -> Union[Tuple[pt.Tensor, pt.Tensor], Tuple[pt.Tensor, List[dict]]]:
        if self.__type == 'train' or self.__type == 'val':
            return self.__collate_labels(batch)
        if self.__type == 'predict':
            return self.__collate_predict(batch)

    def __collate_labels(self, batch: List[dict]) -> Tuple[pt.Tensor, pt.Tensor]:
        """
            Receives a list of QuackImageData with B elements.  Loads pixel data
            from each into a tensor, transforms the tensor for training and stacks
            the batch. Stacks label values into a second tensor.
            Returns a tuple with (B, 3, H, W) shaped tensor and (B, 1) shaped tensor.

            Parameters
            ----------
            batch: List[QuackImageData]
                A list of QuackImageData (TypedDict) in the batch.

            Returns
            -------
            Tuple[pt.Tensor, pt.Tensor]
                A tuple in which the first tensor is batch image data and the second is labels.
        """
        data = []
        labels = []
        for item in batch:
            image = pt.from_numpy(item['pixels'])
            image = self.__transforms(image)
            data.append(image)
            censored = 0
            if item['metadata']['censored'] == 1:
                censored = 1
            labels.append(censored)
        return pt.stack(data), pt.tensor(labels)

    def __collate_predict(self, batch: List[dict]) -> Tuple[pt.Tensor, List[dict]]:
        """
            Receives a list of QuackImageData with B elements.  Loads pixel data  Returns a (B x T) shaped tensor.

            Parameters
            ----------
            batch: List[TokenizedQuackData]
                A list of TokenizedQuackData (TypedDict) in the batch.

            Returns
            -------
            pt.Tensor
        """
        data = []
        meta = []
        for item in batch:
            image = pt.from_numpy(item['pixels'])
            image = self.__transforms(image)
            data.append(image)
            meta.append(item['metadata'])
        return pt.stack(data), meta


class QuackTokenizedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 64, workers: int = 0):
        self.__batch_size = batch_size
        self.__workers = workers
        dataset = QuackImageDataset(data_dir)
        self.__predict_data = QuackImageDataset(data_dir)
        print(f'Source dataset ready with {len(dataset)} items.')
        self.__width = dataset.data_width()
        # Reserve 20% of the data as test data.
        test_reserve = round(len(dataset) * 0.2)
        # Reserve 10% of the data as validation data.
        val_reserve = round(len(dataset) * 0.1)
        self.__train_data, self.__test_data, self.__val_data = random_split(
            dataset, [len(dataset) - test_reserve - val_reserve, test_reserve, val_reserve]
        )
        print(f'Training dataset randomly split with {len(self.__train_data)} items.')
        print(f'Test dataset randomly split with {len(self.__test_data)} items.')
        print(f'Validation dataset randomly split with {len(self.__val_data)} items.')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_collate = QuackImageTransformer(step_type='train')
        return DataLoader(
            self.__train_data,
            batch_size=self.__batch_size,
            collate_fn=train_collate,
            shuffle=True,
            num_workers=self.__workers,
            persistent_workers=True
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_collate = QuackImageTransformer(step_type='val')
        return DataLoader(
            self.__test_data,
            batch_size=self.__batch_size,
            collate_fn=test_collate,
            num_workers=self.__workers,
            persistent_workers=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_collate = QuackImageTransformer(step_type='val')
        return DataLoader(
            self.__val_data,
            batch_size=self.__batch_size,
            collate_fn=val_collate,
            num_workers=self.__workers,
            persistent_workers=True
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        predict_collate = QuackImageTransformer(step_type='predict')
        return DataLoader(
            self.__predict_data,
            batch_size=self.__batch_size,
            collate_fn=predict_collate,
            num_workers=self.__workers,
            persistent_workers=True
        )

    def get_width(self):
        return self.__width