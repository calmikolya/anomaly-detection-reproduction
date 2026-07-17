# Anomaly Detection — воспроизведение 5 моделей на MVTec AD

Воспроизведение пяти методов anomaly detection на датасете MVTec AD (класс `bottle`):
**PatchCore, PaDiM, STFPM, SimpleNet, DRAEM**. Модели используют разные подходы —
сравнение с эталонными признаками, статистическое моделирование, дистилляция
teacher-student, синтетические аномалии в пространстве признаков и на уровне пикселей.
Каждая модель запущена как есть, без переписывания архитектуры; цели — получить
реальные метрики и сравнить их с результатами из оригинальных статей. Среда —
Mac без CUDA (CPU-only).

## Структура репозитория

Каждая модель — клон оригинального репозитория автора с точечными правками под
CPU/macOS. Общего раннера для моделей нет: у каждой свой способ запуска, как в
оригинале.

```
.
├── results.md                  # метрики всех 5 моделей + сравнение с оригинальными статьями
├── patchcore-inspection/        # 1. PatchCore (amazon-science/patchcore-inspection)
├── padim_run/                   # 2. PaDiM (через библиотеку anomalib)
├── STFPM/                       # 3. STFPM (gdwang08/STFPM)
├── SimpleNet/                    # 4. SimpleNet (DonaldRR/SimpleNet)
└── DRAEM/                        # 5. DRAEM (VitjanZ/DRAEM)
```

## Conda-окружения

Вместо пяти изолированных окружений (диска было жалко) обошлись двумя общими:

- **`patchcore`** — для PatchCore, STFPM, SimpleNet и DRAEM
- **`anomalib`** — отдельно для PaDiM, потому что у библиотеки `anomalib` своё большое
  дерево зависимостей (pytorch-lightning, kornia и т.д.), которое проще не смешивать
  с остальным

```bash
conda activate patchcore   # для моделей 1, 3, 4, 5
conda activate anomalib    # для модели 2
```

## Результаты

Все метрики (image-AUROC, pixel-AUROC и сравнение с цифрами из оригинальных статей) —
в **[results.md](results.md)**.
