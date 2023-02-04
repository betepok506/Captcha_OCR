import os
from torch import nn
import torch
from torchmetrics import CharErrorRate
from torch.utils.data import DataLoader
from src.utils import decode_batch_outputs
from sklearn.preprocessing import LabelEncoder
import numpy as np


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.dropout_percentage = 0.5
        self.conv_layers = nn.Sequential(
            # BLOCK-1 (starting block) input=(224x224) output=(56x56)
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),

            # BLOCK-2 (1) input=(56x56) output = (56x56)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.Dropout(p=self.dropout_percentage),
        )

        self.linear_1 = nn.Linear(832, 128)
        self.lstm = nn.GRU(128, 32, bidirectional=True, batch_first=True)
        self.linear_2 = nn.Linear(64, 20)

    def forward(self, x, targets):
        bs, _, _, _ = x.size()
        x = self.conv_layers(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1)
        x = self.linear_1(x)
        x = nn.functional.relu(x)
        x, h = self.lstm(x)
        x = self.linear_2(x)
        x = x.permute(1, 0, 2)
        if targets is not None:
            log_probs = nn.functional.log_softmax(x, 2)

            input_lengths = torch.full(size=(bs,), fill_value=log_probs.size(0),
                                       dtype=torch.int32)

            target_lengths = torch.full(size=(bs,), fill_value=targets.size(1),
                                        dtype=torch.int32)

            loss = nn.CTCLoss(blank=19)(log_probs, targets, input_lengths, target_lengths)

            return x, loss

        return x, None
    # def __init__(self):
    #     super(FCNN_LSTM, self).__init__()
    #     self.conv_layers = nn.Sequential(
    #         nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2),
    #
    #         nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=(2, 8)),
    #         #
    #         # nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
    #         # nn.ReLU(),
    #         # nn.MaxPool2d(kernel_size=(8, 2)),
    #         #
    #         # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
    #         # nn.ReLU(),
    #         # nn.MaxPool2d(kernel_size=(2, 8))
    #     )
    #     self.linear_1 = nn.Linear(768, 128)
    #     self.lstm = nn.GRU(128,
    #                         32,
    #                         bidirectional=True,
    #                         batch_first=True)
    #
    #     self.linear_2 = nn.Linear(64, 20)
    #
    # def forward(self, x, targets):
    #     bs, _, _, _ = x.size()
    #     x = self.conv_layers(x)
    #     x = x.permute(0, 3, 1, 2)
    #     x = x.view(bs, x.size(1), -1)
    #     x = self.linear_1(x)
    #     x = nn.functional.relu(x)
    #     x, h = self.lstm(x)
    #     x = self.linear_2(x)
    #     x = x.permute(1, 0, 2)
    #     if targets is not None:
    #         log_probs = nn.functional.log_softmax(x, 2)
    #
    #         input_lengths = torch.full(size=(bs,), fill_value=log_probs.size(0),
    #                                    dtype=torch.int32)
    #
    #         target_lengths = torch.full(size=(bs,), fill_value=targets.size(1),
    #                                     dtype=torch.int32)
    #
    #         loss = nn.CTCLoss(blank=19)(log_probs, targets, input_lengths, target_lengths)
    #
    #         return x, loss
    #
    #     return x, None


class OCR:
    """
    Класс содержит реализация модели для распознования изображений с капчей
    """

    def __init__(self):
        self.model = CNN_LSTM()
        self._device = "cpu"
        self._epoch = 0
        self._eval_loss = float("inf")
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=3E-4)
        self._path_save_checkpoint = "./model/checkpoints/"

    def train(self,
              train_loader,
              test_loader,
              encoder,
              num_epochs=500,
              save_checkpoint=True,
              visualize_learning=True,
              visualize_each=5):
        """
        Функция для обучения модели

        Parameters
        ------------
        train_loader: `DataLoader`
            Загрузчки данных для обучения
        test_loader: `DataLoader`
            Загрузчки данных для валидации
        encoder: `LabelEncoder`
            Кодировщик меток
        num_epochs: `int`
            Количество эпох обучения
        save_checkpoint: `bool`
            True если необходимо сохранять наилучщую модель при  обучении, иначе False
        visualize_learning: `bool`
            True если необходимо печатать процесс обучения, иначе False
        visualize_each: `int`
            Отвечает через сколько эпох печатать результат обучения

        Returns
        ------------
        `list`, `list`
            Массив потерь при обучении, массив потерь при валидации
        """
        training_loss = []
        evaluations_loss = []
        for epoch in range(num_epochs):
            train_loss = self._train_epoch(train_loader)
            eval_loss, outputs = self._validation_epoch(test_loader)
            training_loss.append(training_loss)
            evaluations_loss.append(eval_loss)
            self._epoch += 1

            if (epoch + 1) % visualize_each == 0:
                print(f"Epoch: {self._epoch} Train loss: {train_loss} Validation loss: {eval_loss}")

                if visualize_learning:
                    all_predictions = []
                    pred_labels = []
                    for e in outputs:
                        batch_predictions_labels, batch_predictions = decode_batch_outputs(e, encoder)
                        all_predictions.extend(batch_predictions)
                        pred_labels.extend(batch_predictions_labels)

                    test_loader_labels = []
                    for images, labels in test_loader:
                        for e in labels:
                            e = e.type(torch.int).tolist()
                            test_label_in_characters = encoder.inverse_transform(e)
                            test_label_original = ''.join(test_label_in_characters)
                            test_loader_labels.append(test_label_original)

                    index = np.random.choice(len(test_loader_labels), 5, replace=False)
                    examples = list(zip([test_loader_labels[ind] for ind in index],
                                        [all_predictions[ind] for ind in index]))
                    print(examples)
                    cer = self._evaluations_cer(test_loader_labels, pred_labels).item()
                    print(f"CER: {cer}")

            if save_checkpoint and self._eval_loss > eval_loss:
                self._eval_loss = eval_loss
                self.save(os.path.join(self._path_save_checkpoint,
                                       f"Epoch_{self._epoch}_loss_{self._eval_loss:.5f}.pt"),
                          True)

        return training_loss, evaluations_loss

    def _train_epoch(self, train_loader: DataLoader):
        """
        Функция для тренировки модели на наборе данных

        Parameters
        ------------
        train_loader: `DataLoader`
            Загрузчки набора данных

        Returns
        ------------
        `float`
            Ошибка при обучении
        """
        self.model.train()
        final_loss = 0
        for images, texts in train_loader:
            self._optimizer.zero_grad()
            images = images.to(self._device)
            targets = texts.to(self._device)
            output, loss = self.model(images, targets)
            loss.requres_grad = True
            loss.backward()
            self._optimizer.step()
            final_loss += loss.item()
            loss.detach()

        train_loss = final_loss / len(train_loader)
        return train_loss

    def _validation_epoch(self, test_loader):
        """
        Функция для валидации модели на наборе данных

        Parameters
        ------------
        test_loader: `DataLoader`
            Загрузчки набора данных

        Returns
        ------------
        `float`
            Ошибка при валидации
        """

        self.model.eval()
        final_loss = 0
        outputs = []
        with torch.no_grad():
            for images, texts in test_loader:
                images = images.to(self._device)
                targets = texts.to(self._device)
                batch_outputs, loss = self.model(images, targets)
                loss.requres_grad = True
                final_loss += loss.item()

                outputs.append(batch_outputs.detach())

        eval_loss = final_loss / len(test_loader)
        return eval_loss, outputs

    def to(self, device: str = "cpu"):
        """
        Функция для задачи устройства для обучения модели

        Parameters
        ------------
        device: `str`
            Устройство для обучения модели
        """
        self._device = device
        self.model.to(device)

    def evaluations(self,
                    test_loader: DataLoader,
                    encoder: LabelEncoder,
                    each_image: bool = False):
        """
        Функция для оценки модели по метрике CharErrorRate

        Parameters
        ------------
        test_loader: `DataLoader`
            Загрузчки набора данных для тестирования
        encoder: `LabelEncoder`
            Кодировщик меток

        Returns
        ------------
        `float`
            Показатель оценки по метрике CharErrorRate
        """
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for images, texts in test_loader:
                images = images.to(self._device)
                targets = texts.to(self._device)
                batch_outputs, loss = self.model(images, targets)
                outputs.append(batch_outputs.detach())

        pred_labels = []
        for e in outputs:
            batch_predictions_labels, _ = decode_batch_outputs(e, encoder)
            pred_labels.extend(batch_predictions_labels)

        test_loader_labels = []
        test_loader_img = []
        for images, labels in test_loader:
            for ind, label in enumerate(labels):
                label = label.type(torch.int).tolist()
                test_label_in_characters = encoder.inverse_transform(label)
                test_label_original = ''.join(test_label_in_characters)
                test_loader_labels.append(test_label_original)
                test_loader_img.append(images[ind].squeeze())

        if each_image:
            cer = []
            for ind in range(len(test_loader_labels)):
                img = test_loader_img[ind]
                cer.append({"img": img,
                            "CER": self._evaluations_cer(test_loader_labels[ind],
                                                         pred_labels[ind]).item(),
                            "true_label": test_loader_labels[ind],
                            "pred_label": pred_labels[ind]})
        else:
            cer = self._evaluations_cer(test_loader_labels, pred_labels).item()

        return cer

    def _evaluations_cer(self, labels_true, labels_pred):
        """
        Функция для оценки строк или набора строк по метрике CharErrorRate

        Parameters
        ------------
        labels_true: `list[str]` or `str`
            Массив истинных меток
        labels_pred: `list[str]` or `str`
            Массив предсказанных меток

        Returns
        ------------
        `float`
            Показатель оценки по метрике CharErrorRate
        """
        cer = CharErrorRate()
        return cer(labels_true, labels_pred)

    def error_calculation_each_image(self, test_loader):
        pass

    def save(self, path_to_save: str, training: bool = False):
        """
        Функция для сохранения модели

        Parameters
        ------------
        path_to_save: `str`
            Путь, куда будет сохранена модель
        training: `bool`
            Флаг означающий стоит ли сохранять информацию, необходимую для продолжения обучения
        """
        path_to_folder = os.path.split(path_to_save)
        if len(path_to_folder) == 2:
            os.makedirs(path_to_folder[0], exist_ok=True)
        state = {
            'model_state_dict': self.model.state_dict()
        }
        if training:
            state['optimizer_state_dic'] = self._optimizer.state_dict()
            state['epoch'] = self._epoch
            state['loss'] = self._eval_loss

        torch.save(state, path_to_save)

    def load(self, path_to_model):
        """
        Функция для загрузки модели

        Parameters
        ------------
        path_to_model: `str`
            Путь до загружаемое модели
        """
        checkpoint = torch.load(path_to_model)
        if "optimizer_state_dic" in checkpoint:
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dic"])

        if "epoch" in checkpoint:
            self._epoch = checkpoint["epoch"]

        if "epoch" in checkpoint:
            self._eval_loss = checkpoint["loss"]

        self.model.load_state_dict(checkpoint["model_state_dict"])