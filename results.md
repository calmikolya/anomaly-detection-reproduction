# Результаты воспроизведения 5 моделей anomaly detection

Класс MVTec AD: **bottle** (209 train / 83 test изображений). Все модели обучены и
провалидированы на CPU (Apple Silicon, без CUDA).

Контрольные значения ниже — усреднённые по всем 15 классам MVTec AD из оригинальных
статей. Наши цифры — только по классу `bottle`, поэтому расхождение в 1-3 п.п.
в любую сторону — это ожидаемо и нормально (bottle — один из "лёгких" классов, часто
даёт результат выше среднего по датасету).

| Модель | image-AUROC (bottle) | pixel-AUROC (bottle) | Контроль image-AUROC (статья, весь MVTec) | Контроль pixel-AUROC (статья, весь MVTec) |
|---|---|---|---|---|
| PatchCore | 1.000 | 0.985 | ~99.6% | ~98.4% |
| PaDiM | 0.998 | 0.981 | — | ~97.9% |
| STFPM | 1.000 | 0.989 | — | ~97.0% |
| SimpleNet | 1.000 | 0.978 | ~99.6% | — |
| DRAEM | TBD | TBD | ~98.1% | ~97.5% |

Доп. метрики: STFPM PRO = 0.961, SimpleNet anomaly_pixel_auroc (PRO-подобная метрика) = 0.913
(per-region overlap, тоже считаются репозиториями).

## Детали по моделям

### 1. PatchCore
- Репозиторий: `patchcore-inspection/` (склонирован заранее, официальный amazon-science)
- Конфиг: WideResNet50, слои layer2+layer3, coreset 10%, embed dim 1024, patchsize 3, IM224 baseline
- Обучение = построение memory bank (без градиентного спуска), инференс = поиск ближайших соседей (FAISS CPU)
- Команда: см. `patchcore-inspection/README.md`, наш запуск — ниже в этом файле

### 2. PaDiM
- Через библиотеку `anomalib` (conda env `anomalib`)
- Backbone resnet18, слои layer1+layer2+layer3, 1 "эпоха" (Gaussian fitting, не градиентное обучение)
- Скрипт: `padim_run/run_padim.py`

### 3. STFPM
- Репозиторий: `STFPM/` (gdwang08/STFPM)
- ResNet18 teacher-student, 100 эпох (вместо 200 в примере авторов — сокращено для разумного времени на CPU)
- Адаптации: `.cuda()` → `.to(device)`, `np.bool` → `bool` (numpy 2.x)

### 4. SimpleNet
- Репозиторий: `SimpleNet/` (DonaldRR/SimpleNet)
- WideResNet50, 10 мета-эпох x 4 gan-эпохи (вместо 40x4 — сокращено для CPU из-за дедлайна, после 2 неудачных попыток на 15), imagesize 224 (вместо 288)
- Адаптации: `--gpu` default → CPU, `prefetch_factor` фикс для `num_workers=0`, `pandas.DataFrame.append` → `pd.concat`-совместимый код, `np.bool` → `bool`

### 5. DRAEM
- Репозиторий: `DRAEM/` (VitjanZ/DRAEM)
- Требует датасет текстур DTD (скачан отдельно, `dtd/`) для генерации синтетических аномалий
- Эпохи: TBD (вместо 700 в статье — сокращено для CPU)
- Адаптации: `.cuda()` → `.to(device)`, `num_workers=16` → `0`, фикс `imgaug` под numpy 2.x (`np.sctypes` removed)

## Как воспроизвести

См. корневой `README.md` — там пошагово расписано, какое conda-окружение активировать
и какую команду запускать для каждой модели.

## Общие проблемы при запуске на Mac (CPU) и их решения

1. **Датасет MVTec AD** — не входит в репозиторий, скачивается отдельно с официального сайта.
2. **Крах с SIGSEGV внутри libomp** (PatchCore) — конфликт двух пулов потоков OpenMP
   (PyTorch и FAISS одновременно). Решение: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
   VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1` при запуске PatchCore.
   Для остальных моделей (без FAISS) это не требуется — там работает многопоточность.
3. **Краш DataLoader (multiprocessing/fork) на macOS** — решение: `num_workers=0` везде.
4. **`np.bool`, `np.sctypes` удалены в numpy 2.x** — точечные правки в клонированных
   репозиториях (см. пункт "Адаптации" по каждой модели).
5. **`pandas.DataFrame.append` удалён в pandas 2.x** — заменено на построение списка строк
   + `pd.DataFrame(rows)`.
