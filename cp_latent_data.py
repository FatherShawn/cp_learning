from typing import Tuple, List, Union
import torch as pt
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
from cp_latent_dataset import QuackLatentData, QuackLatentDataset


class QuackLatentCollator:

    def __init__(self, step: str, strategy: str) -> None:
        super().__init__()
        valid_steps = {'train', 'val', 'test', 'predict'}
        if step not in valid_steps:
            raise ValueError('Improper transform step.')
        self.__step = step

    def __call__(self, batch: List[dict], *args, **kwargs) -> Union[Tuple[pt.Tensor, pt.Tensor], Tuple[pt.Tensor, List[dict]]]:
        if self.__step == 'predict':
            return self.__collate_predict(batch)
        return self.__collate_labels(batch)

    def __collate_labels(self, batch: List[dict]) -> Tuple[pt.Tensor, pt.Tensor]:
        """
            Receives a list of QuackLatentData with B elements.  Loads encoded data
            from each into a tensor. Stacks label values into a second tensor.
            Returns a tuple with (B, H, W) shaped tensor and (B, 1) shaped tensor.

            Parameters
            ----------
            batch: List[QuackLatentData]
                A list of QuackLatentData (TypedDict) in the batch.

            Returns
            -------
            Tuple[pt.Tensor, pt.Tensor]
                A tuple in which the first tensor is batch image data and the second is labels.
        """
        data = []
        labels = []
        for item in batch:
            data.append(pt.from_numpy(item['encoded']))
            censored = pt.tensor([0]).to(pt.float)
            if item['metadata']['censored'] == 1:
                censored = pt.tensor([1]).to(pt.float)
            labels.append(censored)
        return pt.stack(data), pt.stack(labels)

    def __collate_predict(self, batch: List[dict]) -> Tuple[pt.Tensor, List[dict]]:
        """
            Receives a list of QuackImageData with B elements.  Loads pixel data  Returns a (B x T) shaped tensor.

            Parameters
            ----------
            batch: List[TokenizedQuackData]
                A list of QuackLatentData (TypedDict) in the batch.

            Returns
            -------
            pt.Tensor
        """
        data = []
        meta = []
        for item in batch:
            data.append(pt.from_numpy(item['encoded']))
            meta.append(item['metadata'])
        return pt.stack(data), meta


class QuackLatentDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 64, workers: int = 0):
        self.__batch_size = batch_size
        self.__workers = workers
        dataset = QuackLatentDataset(data_dir)
        self.__predict_data = dataset
        print(f'Source dataset ready with {len(dataset)} items.')
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
        print(f'Prediction dataset ready with {len(self.__predict_data)} items.')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_collate = QuackLatentCollator(step='train')
        return DataLoader(
            self.__train_data,
            batch_size=self.__batch_size,
            collate_fn=train_collate,
            shuffle=True,
            num_workers=self.__workers,
            persistent_workers=True
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_collate = QuackLatentCollator(step='test')
        return DataLoader(
            self.__test_data,
            batch_size=self.__batch_size,
            collate_fn=test_collate,
            num_workers=self.__workers,
            persistent_workers=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_collate = QuackLatentCollator(step='val')
        return DataLoader(
            self.__val_data,
            batch_size=self.__batch_size,
            collate_fn=val_collate,
            num_workers=self.__workers,
            persistent_workers=True
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        predict_collate = QuackLatentCollator(step='predict')
        return DataLoader(
            self.__predict_data,
            batch_size=self.__batch_size,
            collate_fn=predict_collate,
            num_workers=self.__workers,
            persistent_workers=True
        )