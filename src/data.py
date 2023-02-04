import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from skimage import io
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os


def show_landmarks(image):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.pause(0.001)  # pause a bit so that plots are updated


class CaptchaDataset(Dataset):
    """Класс содержит собственную реализацию Dataset унаследованную от `torch.utils.data.Dataset`"""

    def __init__(self, paths_to_images, targets_encoded, transforms=None):
        self.paths_to_images = paths_to_images
        self.targets = targets_encoded
        self.transform = transforms

    def __len__(self):
        return len(self.paths_to_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.paths_to_images[idx]
        image = io.imread(img_name)

        target = self.targets[idx]
        tensorized_target = torch.tensor(target, dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        return image, tensorized_target


def create_loaders(paths_to_images: list,
                   labels: list,
                   transform: transforms,
                   batch_size: int = 16,
                   test_size: float=0.2):
    """
    Функция реализует создание наборов данных

    Parameters
    ------------
    paths_to_images: `list`
        Массив, содержащий пути до изображений
    labels: `np.array`
        Массив, содержащий закодированные метки
    transform: `transforms`
        Преобразования, которые необходимо применить к данным
    batch_size: `int`
        Размер батча
    test_size: `float`
        Размер тестовой части

    Returns
    ------------
    `DataLoader`, `DataLoader`
        Загрузчик данных обучающей части датасета, загрузчик данных тестовой части датасета
    """

    train_img, test_img, train_targets, test_targets = train_test_split(paths_to_images,
                                                                        labels,
                                                                        test_size=test_size,
                                                                        random_state=7)

    train_dataset = CaptchaDataset(train_img, train_targets, transforms=transform)
    test_dataset = CaptchaDataset(test_img, test_targets, transforms=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def extract_data(path_to_file: str):
    """
    Функция для чтения файла аннотации и преобразования данных в формат датасета.
    Для кодирования меток используется `LabelEncoder`

    Parameters
    ------------
    path_to_file: `str`
        Путь до файла с аннотацией

    Returns
    ------------
    `list`, `np.array`, `LabelEncoder`
        Массив путей до изображений, массив закодированных меток, кодировщик меток
    """

    annotations = pd.read_csv(path_to_file)
    paths_to_images = annotations.iloc[:, 0].tolist()
    targets_orig = annotations.iloc[:, 1]

    labels = targets_orig.tolist()
    targets = [[c for c in x] for x in labels]
    targets_flat = [c for clist in targets for c in clist]

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(targets_flat)
    targets_enc = [label_encoder.transform(x) for x in targets]
    targets_enc = np.array(targets_enc)

    return paths_to_images, targets_enc, label_encoder


def create_annotations(path_to_dataset: str, path_to_save: str):
    """
    Функция для создания файла аннотации

    Parameters
    ------------
    path_to_dataset: `str`
        Путь к папке с изображениями
    path_to_save: `str`
        Путь, где будет сохранен файл с аннотацией
    """
    paths_to_images = []
    for file_name in os.listdir(path_to_dataset):
        if len(file_name.split('.')) == 2:
            decoding = file_name.split('.')[0]
            paths_to_images.append([os.path.join(path_to_dataset, file_name), decoding])

    pd.DataFrame(data=paths_to_images).to_csv(path_to_save, index=False, header=False)
