#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль для анализа тональности мнений пользователей Интернет-ресурсов:
сравнение Baseline и BiLSTM.

Реализует полный цикл обработки текстов (на примере датасета
RuReviews, 3 класса):
от загрузки данных и токенизации
(RuBERT Tokenizer) до обучения и оценки моделей.

В модуле сравниваются две архитектуры на базе PyTorch:
    - Baseline: Embedding -> Average Pooling -> FC (игнорирует порядок слов).
    - BiLSTM: Embedding -> Bidirectional LSTM -> FC (учитывает контекст).

Включает замер времени обучения, сравнение по метрике Accuracy
(с целевым порогом 70%)
и визуализацию графиков Loss/Accuracy.

Точка входа и общая логика запуска: смотрите функцию main().
"""

import time

import matplotlib.pyplot as plt
import pandas as pd
import requests
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer

# --- Конфигурации ---
BATCH_SIZE = 32  # Размер batch (баланс скорости и памяти GPU)
MAX_LEN = 128  # Максимальная длина последовательности в токенах
EPOCHS = 5  # Количество эпох обучения
EMBED_DIM = 128  # Размерность embedding вектора
HIDDEN_DIM = 64  # Размер hidden state LSTM
NUM_CLASSES = 3  # Количество классов (negative, neutral, positive)
SAMPLE_SIZE = 30000  # Ограничение выборки для ускорения обучения
DATA_URL = (
    "https://github.com/sismetanin/rureviews/raw/refs/heads/"
    "master/women-clothing-accessories.3-class.balanced.csv"
)


def format_time(seconds):
    """Форматирование времени в часы/минуты/секунды."""
    secs = int(seconds % 60)
    mins = int((seconds // 60) % 60)
    hrs = int(seconds // 3600)

    if hrs > 0:
        return f"{hrs} ч. {mins} мин. {secs} сек."
    if mins > 0:
        return f"{mins} мин. {secs} сек."
    return f"{secs} сек."


def download_and_prepare_data():
    """Загрузка, очистка и разделение данных."""
    print("\n" + "=" * 50)
    print("ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    print("=" * 50)
    print(f"Целевой размер выборки: {SAMPLE_SIZE} строк")

    response = requests.get(DATA_URL, timeout=30)
    with open("data.csv", "wb") as f:
        f.write(response.content)

    # sep=None заставляет pandas автоматически определить разделитель
    # noinspection PyArgumentList
    df = pd.read_csv(
        "data.csv", sep=None, engine='python',
        on_bad_lines='skip'  # Пропускаем строки с ошибками
    )
    # noinspection PyUnresolvedReferences
    df = df[['review', 'sentiment']].rename(
        columns={'review': 'text'}
    )
    # Удаляем пустые значения
    df = df.dropna(subset=['text', 'sentiment'])
    # Удаляем дублированные значения
    df = df.drop_duplicates(subset=['text'])

    # Преобразуем текстовые метки в числовой формат 0, 1, 2
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label'] = df['sentiment'].map(label_map)
    # Удаляем строки, если встретились неизвестные метки
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    if len(df) > SAMPLE_SIZE:
        df = df.sample(
            n=SAMPLE_SIZE, random_state=42
        ).reset_index(drop=True)

    # stratify сохраняет пропорции классов в train и val выборках
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    print(
        f"Данные готовы. Train: {len(train_df)}, "
        f"Val: {len(val_df)}"
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def prepare_dataloaders(train_df, val_df, tokenizer):
    """Токенизация и создание DataLoader."""
    train_texts = train_df['text'].astype(str).tolist()
    val_texts = val_df['text'].astype(str).tolist()

    # padding='max_length' дополняет тексты нулями до MAX_LEN,
    # чтобы собрать последовательности разной длины в единый тензор
    train_enc = tokenizer(
        train_texts, add_special_tokens=True,
        max_length=MAX_LEN, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    val_enc = tokenizer(
        val_texts, add_special_tokens=True,
        max_length=MAX_LEN, padding='max_length',
        truncation=True, return_tensors='pt'
    )

    # TensorDataset связывает input_ids и labels по индексу
    train_dataset = TensorDataset(
        train_enc['input_ids'],
        torch.tensor(train_df['label'].values, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        val_enc['input_ids'],
        torch.tensor(val_df['label'].values, dtype=torch.long)
    )

    # shuffle=True важен для обучения, чтобы модель не запоминала порядок
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    return train_loader, val_loader


class BaseSentimentModel(nn.Module):
    """Базовый класс для моделей анализа тональности."""

    def count_parameters(self):
        """Возвращает количество обучаемых параметров."""
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad
        )

    def predict(self, input_ids, device):
        """Возвращает предсказанные классы для batch."""
        self.eval()
        with torch.no_grad():
            outputs = self(input_ids.to(device))
            _, predictions = torch.max(outputs, dim=1)
        return predictions


class BaselineModel(BaseSentimentModel):
    """
    Baseline модель: Эмбеддинг -> Усреднение -> Полносвязный слой.

    Логика:
    1. Эмбеддинг: каждый токен заменяется на вектор из обучаемой таблицы.
    2. Усреднение (Average Pooling): векторы всех токенов предложения
       складываются и делятся на их количество. Порядок слов полностью
       игнорируется, модель видит только "средний" смысл текста.
    3. Полносвязный слой: итоговый усредненный вектор умножается на матрицу
       весов, чтобы получить оценки (логиты) для каждого из классов.
    """

    def __init__(self, vocab_size, embed_dim, num_classes):
        """Инициализация слоев."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        # Dropout отключает 30% нейронов для регуляризации
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """Прямой проход (forward pass)."""
        embedded = self.dropout(self.embedding(x))
        # Усредняем embedding по всей последовательности.
        # Порядок слов полностью игнорируется.
        pooled = embedded.mean(dim=1)
        return self.fc(pooled)


class LSTMClassifier(BaseSentimentModel):
    """
    Архитектура на основе двунаправленной LSTM (BiLSTM).

    Логика прямого прохода:
    1. Эмбеддинг: токены превращаются в векторы.
    2. LSTM-ячейка: сеть читает текст шаг за шагом. На каждом шаге она решает:
       - что забыть из прошлой памяти;
       - какую новую информацию добавить из текущего слова;
       - какую часть памяти передать на следующий шаг.
    3. Двунаправленность (BiLSTM): текст читается дважды — от начала к концу
       и от конца к началу. Последние скрытые состояния обоих направлений
       склеиваются в один длинный вектор, содержащий контекст всего предложения.
    4. Полносвязный слой: склеенный вектор проходит через линейный слой
       для получения оценок по классам.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        """Инициализация слоев BiLSTM."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # batch_first=True: вход имеет форму (batch, seq_len, embed_dim)
        # bidirectional=True: читает текст слева направо и наоборот
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True,
            bidirectional=True
        )
        # Умножаем на 2, так как склеиваем выходы двух направлений
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """Прямой проход (forward pass) BiLSTM."""
        embedded = self.dropout(self.embedding(x))

        # LSTM возвращает все скрытые состояния и кортеж
        # (h_n, c_n) — последние состояния для каждого слоя
        _, (hidden, _) = self.lstm(embedded)

        # Форма hidden: (num_layers * num_directions, batch, hidden_dim)
        # hidden[-2] — последний hidden state прямого прохода
        # hidden[-1] — последний hidden state обратного прохода
        # Склеиваем их для получения полного контекста предложения
        hidden = torch.cat(
            (hidden[-2, :, :], hidden[-1, :, :]), dim=1
        )
        return self.fc(self.dropout(hidden))


class TrainConfig:
    """Конфигурация для процесса обучения."""

    def __init__(self, device, lr, epochs, model_name):
        """Инициализация конфигурации."""
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.model_name = model_name

    def display(self, num_params: int):
        """Выводит информацию о конфигурации обучения."""
        print("\n" + "-" * 50)
        print(f"ОБУЧЕНИЕ МОДЕЛИ: {self.model_name}")
        print("-" * 50)
        print(f"Количество параметров: {num_params:,}")
        print(
            f"Оптимизатор: AdamW | Шаг обучения (lr): {self.lr}"
        )
        print("-" * 50)

    def to_dict(self):
        """Возвращает конфигурацию в виде словаря для логирования."""
        return {
            'model_name': self.model_name,
            'lr': self.lr,
            'epochs': self.epochs,
            'device': str(self.device)
        }


class Trainer:
    """Класс для управления циклом обучения модели."""

    def __init__(self, model, loaders, config):
        """
        Инициализирует тренера.

        Args:
            model: Модель PyTorch для обучения.
            loaders: Кортеж (train_loader, val_loader).
            config: Объект конфигурации TrainConfig.
        """
        self.model = model.to(config.device)
        self.train_loader, self.val_loader = loaders
        self.config = config
        self.optimizer = AdamW(
            self.model.parameters(), lr=config.lr
        )
        # CrossEntropyLoss объединяет LogSoftmax и NULLLoss
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'val_acc': []}

    def run(self):
        """Запускает цикл обучения и валидации."""
        self.config.display(self.model.count_parameters())

        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            avg_loss = self.train_epoch()
            val_acc = self.validate()

            self.history['train_loss'].append(avg_loss)
            self.history['val_acc'].append(val_acc)

            duration = time.time() - epoch_start
            msg = (
                f"Эпоха {epoch:02d}/{self.config.epochs} | "
                f"Время: {format_time(duration):>12} | "
                f"Loss: {avg_loss:.4f} | "
                f"Accuracy: {val_acc:.4f}"
            )
            print(msg)

        total_time = time.time() - start_time
        name = self.config.model_name
        print(
            f"Обучение {name} завершено за "
            f"{format_time(total_time)}"
        )
        return self.history

    def train_epoch(self):
        """Выполняет одну эпоху обучения."""
        # Включаем режим обучения (Dropout активен)
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            ids = batch[0].to(self.config.device)
            labels = batch[1].to(self.config.device)

            # Стандартный шаг оптимизации в PyTorch:
            # 1. Сбрасываем градиенты с прошлой итерации
            self.optimizer.zero_grad()
            # 2. Forward pass: получаем предсказания
            outputs = self.model(ids)
            # 3. Вычисляем loss
            loss = self.criterion(outputs, labels)
            # 4. Backward pass: считаем градиенты
            loss.backward()
            # 5. Обновляем веса на основе градиентов
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        """Выполняет валидацию."""
        # Включаем режим оценки (Dropout отключен)
        self.model.eval()
        correct = 0
        total = 0

        # Отключаем вычисление градиентов для экономии памяти
        with torch.no_grad():
            for batch in self.val_loader:
                ids = batch[0].to(self.config.device)
                labels = batch[1].to(self.config.device)

                outputs = self.model(ids)
                _, predictions = torch.max(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return correct / total


def plot_results(history_baseline, history_lstm):
    """Отрисовка и сохранение графиков обучения."""
    epochs = range(1, EPOCHS + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs, history_baseline['train_loss'],
        'r-o', label='Baseline (Loss)'
    )
    plt.plot(
        epochs, history_lstm['train_loss'],
        'b-s', label='BiLSTM (Loss)'
    )
    plt.title('Функция потерь (Loss) на обучении')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('loss_plot.png')
    print("График Loss сохранен: loss_plot.png")

    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs, history_baseline['val_acc'],
        'r-o', label='Baseline (Accuracy)'
    )
    plt.plot(
        epochs, history_lstm['val_acc'],
        'b-s', label='BiLSTM (Accuracy)'
    )
    plt.axhline(
        y=0.7, color='g', linestyle='--', label='Цель 70%'
    )
    plt.title('Accuracy на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('accuracy_plot.png')
    print("График Accuracy сохранен: accuracy_plot.png")


def _train_model(model, loaders, config):
    """Создает тренер и запускает обучение."""
    trainer = Trainer(model, loaders, config)
    return trainer.run()


def _print_comparison(history_baseline, history_lstm):
    """Выводит сравнительный анализ двух моделей."""
    b_acc = history_baseline['val_acc'][-1]
    l_acc = history_lstm['val_acc'][-1]

    print("\n" + "=" * 50)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
    print("=" * 50)
    print(f"Baseline Accuracy на валидации: {b_acc:.2%}")
    print(f"BiLSTM   Accuracy на валидации: {l_acc:.2%}")
    print("-" * 50)

    if l_acc > b_acc:
        diff = l_acc - b_acc
        print(f"BiLSTM превосходит Baseline на {diff:.2%}")
    else:
        print("BiLSTM не улучшила результат Baseline.")

    if l_acc >= 0.7:
        print("Целевой показатель (Accuracy >= 70%) достигнут.")
    else:
        print("Целевой показатель (Accuracy >= 70%) не достигнут.")


def main():
    """Запуск полного цикла обучения и визуализации."""
    start_time = time.time()
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("\n" + "=" * 50)
    print("ЗАПУСК АНАЛИЗА ТОНАЛЬНОСТИ")
    print("=" * 50)
    print(f"Устройство: {device}")

    train_df, val_df = download_and_prepare_data()

    # RuBERT tokenizer использует WordPiece, хорошо работает
    # с русской морфологией и не требует предобработки текста
    tokenizer = BertTokenizer.from_pretrained(
        'blanchefort/rubert-base-cased-sentiment'
    )

    loaders = prepare_dataloaders(train_df, val_df, tokenizer)

    baseline_config = TrainConfig(
        device=device, lr=1e-3, epochs=EPOCHS,
        model_name="Baseline"
    )
    history_baseline = _train_model(
        BaselineModel(
            tokenizer.vocab_size, EMBED_DIM, NUM_CLASSES
        ),
        loaders, baseline_config
    )

    lstm_config = TrainConfig(
        device=device, lr=1e-3, epochs=EPOCHS,
        model_name="BiLSTM"
    )
    history_lstm = _train_model(
        LSTMClassifier(
            tokenizer.vocab_size, EMBED_DIM, HIDDEN_DIM,
            NUM_CLASSES
        ),
        loaders, lstm_config
    )

    _print_comparison(history_baseline, history_lstm)

    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("Выполнение завершено.")
    print(f"Общее время: {format_time(total_time)}")
    print("=" * 50 + "\n")

    plot_results(history_baseline, history_lstm)


if __name__ == '__main__':
    main()
