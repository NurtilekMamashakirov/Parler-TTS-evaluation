# Оценка модели синтеза речи parler-tts-mini-jenny-30H

Данный код будет воспроизводить оценку модели parler-tts-mini-jenny-30H по нескольким метрикам

---

## Содержание

1. [Сценарии оценивания](#сценарии-оценивания)
2. [Выбранные метрики и аудиозаписи](#установка)
3. [Результаты оценивания](#результаты-оценивания)
4. [Зависимости](#зависимости)
5. [Установка среды и программы](#установка-среды-и-программы)

---

## Сценарии оценивания

Предполагается 3 сценария оценивания:

1) Проверка стандартного функционала. Она предполагает, что модель правильно синтезирует слова в речь на уровне
   возможности их разбора.
2) Объективная оценка. В этой секции будем сравнивать аудиозаписи, полученные меделью и целевые на схожесть.
3) Субъективная оценка. Так как нет возможности опросить людей о качестве звука и речи в получаемых моделью аудизаписях,
   воспользуемся одной из синтетических моделей, которые предсказывают оценки людей.

---

## Выбранные метрики и тестовые примеры

1. Проверка стандартного функционала будет выполняться с помощью метрики WER (Word Error Rate). Для этого необходимо
   использовать другую модель, обученную переводить речь в текст. Главным требованием к данной модели является
   надежность и высокая точность перевода, чтобы она не ошибалась на сгенерированных нашей tts аудизаписях и не портила
   оценку. Тексты, на которых мы будем проводить это тестирование, могут быть произвольными, так как оценка будет
   объективней, если модель не видела выбранные фразы в процессе своего обучения.
2. Для объективной оценки используем следующие метрики: PESQ (Perceptual Evaluation of Speech Quality), SI-SDR (
   Scale-Invariant Signal-to-Distortion Ratio), STOI (Short-Time Objective Intelligibility). PESQ измеряет качество
   речи, воспринимаемое человеком, сравнивая оригинальный и обработанный (например, очищенный от шума) сигнал. SI-SDR
   измеряет, насколько хорошо восстановленный сигнал (например, речь) соответствует оригиналу, игнорируя различия в
   масштабе (громкости). STOI измеряет разборчивость речи на основе кратковременного анализа (по фрагментам), сравнивая
   спектры оригинального и обработанного. Размеченные фразы возьмем из датасета Jenny, предположительно модель должна
   показать хорошие значения метрик, так как на этих данных она обучалась (скорее всего, в Jenny нет разбиения на
   тренировочную и тестовую выборку, есть и шанс, что взятые примеры будут из тестовой выборки разработчиков).
3. Субъективная оценка будет проводиться посредством метрики DNSMOS (Deep Noise Suppression Mean Opinion Score).
   Простыми словами, DNSMOS предсказывает, как бы люди оценили работу нашей модели по шкале от 1 до 5.

### Результаты оценивания

- Метрика Word Error Rate на 5 фиксированных примерах из кода:
  ![image](./examples/wer.png)
  Диаграмма показывает, что на всех тестах значение WER очень мало (<10), что является хорошим результатом и говорит о
  малой доли ошибки распознавания слов в синтезированной речи
- Объективные метрики: PESQ, SI-SDR, STOI на 5 рандомных примерах из датасета Jenny:
  ![image](./examples/pesq.png) ![image](./examples/si-sdr.png) ![image](./examples/stoi.png)
  Данные метрики показали низкое значение качества. Лишь PESQ показал примерно средние показатели оценки. Возможно,
  проблема заключается в том, что модель использует частоту дискретизации 44100 Гц, а в датасете все аудио-файлы с
  частотой 48000 Гц
- Метрика Deep Noise Suppression Mean Opinion Score на 5 фиксированных примерах из кода:
  ![image](./examples/dnsmos.png)
  Значения оченки качества сигнала (Signal Quality) варируется в диапазон 3.6 - 3.8. Значения оценки шумов (Background
  Noise) во всех тестах лежит в окрестности 4.2. Лингвистическая оценка (Linguistic) находится в диапазоне от 3.3 до
  3.6. Общая оценка - от 3.9 до 4.3. Данные показатели говорят о том, что генерируемые аудиозаписи хорошо оценились бы
  реальными людьми.

## Зависимости

Для работы проекта необходимы следующие библиотеки и их версии:

- `torch~=2.5.1` — Библиотека для работы с нейронными сетями.
- `soundfile~=0.12.1` — Библиотека для чтения и записи аудиофайлов.
- `transformers~=4.46.1` — Библиотека от Hugging Face для работы с трансформерами.
- `torchmetrics~=1.5.2` — Библиотека метрик для PyTorch.
- `jiwer~=3.0.5` — Инструмент для вычисления метрик ошибок текста, таких как WER (Word Error Rate), который используется
  для оценки качества распознавания речи.
- `pandas~=2.0.3` — Библиотека для анализа и манипуляции данными, особенно табличными данными (DataFrame). Она
  предоставляет удобные структуры данных для обработки больших объемов данных
- `requests~=2.31.0` — Популярная HTTP-библиотека для взаимодействия с веб-сервисами через API. Используется для
  получения и отправки данных через HTTP-запросы.
- `numpy~=1.24.3` — Основная библиотека для научных и численных вычислений. Используется для работы с многомерными
  массивами данных и выполнения математических операций.
- `matplotlib~=3.7.2` — Библиотека для визуализации данных, позволяющая создавать графики, диаграммы и другие виды
  графического представления информации.
- `librosa~=0.10.2.post1` — Библиотека для анализа и обработки аудиосигналов. Предоставляет функции для изменения
  частоты дискретизации, извлечения признаков и других операций с аудиофайлами.

### Установка среды и программы

1. Установка зависимостей:

```bash
pip install -r requirements.txt
```

2. Установка parler-tts-mini-jenny-30H

```bash
pip install git+https://github.com/huggingface/parler-tts.git
```

3. Запуск программы

```bash
python main.py
```