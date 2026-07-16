# Anomaly Detection — воспроizведение 5 моделей на MVTec AD

Воспроизведение 5 популярных методов anomaly detection на датасете **MVTec AD**
(класс `bottle`), адаптированное для запуска на Mac без CUDA (CPU-only).

Итоговые метрики и сравнение с контрольными значениями из статей — см. **[results.md](results.md)**.

## Структура репозитория

Каждая модель — это отдельный клон оригинального репозитория авторов, с точечными
правками под CPU/macOS (без переписывания архитектуры). Общей точки входа (`run.py`)
нет — у каждой модели свой способ запуска, как и в оригинале.

```
.
├── results.md                  # итоговая таблица метрик + пояснения
├── patchcore-inspection/        # 1. PatchCore (amazon-science/patchcore-inspection)
├── padim_run/                   # 2. PaDiM (через библиотеку anomalib)
│   └── run_padim.py
├── STFPM/                       # 3. STFPM (gdwang08/STFPM)
├── SimpleNet/                    # 4. SimpleNet (DonaldRR/SimpleNet)
├── DRAEM/                        # 5. DRAEM (VitjanZ/DRAEM)
├── data/mvtec/                  # датасет MVTec AD (не в git, см. ниже)
└── dtd/                          # датасет DTD для DRAEM (не в git, см. ниже)
```

## Датасеты (не входят в репозиторий)

- **MVTec AD**: скачать с https://www.mvtec.com/company/research/datasets/mvtec-ad,
  распаковать в `data/mvtec/` так, чтобы получилось `data/mvtec/bottle/train/good/...`
- **DTD** (нужен только для DRAEM): https://www.robots.ox.ac.uk/~vgg/data/dtd/,
  распаковать в `dtd/` так, чтобы получилось `dtd/images/<category>/*.jpg`

## Conda-окружения

Чтобы не создавать 5 изолированных окружений (ограничение по месту на диске),
используются два общих:

- **`patchcore`** — PatchCore, STFPM, SimpleNet, DRAEM (torch, torchvision, faiss-cpu,
  timm, opencv-python, imgaug, scikit-image, scikit-learn, tensorboard, pandas)
- **`anomalib`** — PaDiM (через библиотеку anomalib, отдельное окружение из-за большого
  количества пинов зависимостей: pytorch-lightning, kornia, FrEIA и т.д.)

```bash
conda activate patchcore   # для моделей 1, 3, 4, 5
conda activate anomalib    # для модели 2
```

Для моделей, использующих `faiss` (PatchCore), на macOS может понадобиться:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```
(иначе — сегфолт из-за конфликта потоков OpenMP между PyTorch и FAISS). Для остальных
моделей ограничение потоков не нужно — только `KMP_DUPLICATE_LIB_OK=TRUE` и `num_workers=0`
в датасетах (multiprocessing-краши на macOS).

## Как запустить каждую модель

### 1. PatchCore
```bash
conda activate patchcore
cd patchcore-inspection
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src python bin/run_patchcore.py --seed 0 --save_patchcore_model \
  --log_group IM224_WR50_bottle --log_project MVTecAD_Results results_bottle \
  patch_core -b wideresnet50 -le layer2 -le layer3 \
  --pretrain_embed_dimension 1024 --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
  sampler -p 0.1 approx_greedy_coreset \
  dataset --num_workers 0 --resize 256 --imagesize 224 -d bottle mvtec ../data/mvtec
```

### 2. PaDiM (anomalib)
```bash
conda activate anomalib
cd padim_run
KMP_DUPLICATE_LIB_OK=TRUE python run_padim.py
```

### 3. STFPM
```bash
conda activate patchcore
cd STFPM
KMP_DUPLICATE_LIB_OK=TRUE python main.py train --mvtec-ad ../data/mvtec --category bottle --epochs 100 --model-save-path snapshots
KMP_DUPLICATE_LIB_OK=TRUE python main.py test --mvtec-ad ../data/mvtec --category bottle --checkpoint snapshots/bottle/best.pth.tar
```

### 4. SimpleNet
```bash
conda activate patchcore
cd SimpleNet
KMP_DUPLICATE_LIB_OK=TRUE python main.py --seed 0 --log_group simplenet_bottle --log_project MVTecAD_Results \
  --results_path results_bottle --run_name run \
  net -b wideresnet50 -le layer2 -le layer3 --pretrain_embed_dimension 1536 --target_embed_dimension 1536 \
  --patchsize 3 --meta_epochs 15 --embedding_size 256 --gan_epochs 4 --noise_std 0.015 \
  --dsc_hidden 1024 --dsc_layers 2 --dsc_margin .5 --pre_proj 1 \
  dataset --num_workers 0 --batch_size 8 --resize 256 --imagesize 224 -d bottle mvtec ../data/mvtec
```

### 5. DRAEM
```bash
conda activate patchcore
cd DRAEM
KMP_DUPLICATE_LIB_OK=TRUE python train_DRAEM.py --obj_id 1 --bs 8 --lr 0.0001 --epochs <N> \
  --data_path ../data/mvtec/ --anomaly_source_path ../dtd/images \
  --checkpoint_path checkpoints --log_path logs
# тестирование — см. results.md, тестировался только bottle через отдельный скрипт
```

## Основные адаптации под Mac/CPU (без переписывания архитектуры)

См. подробности и объяснение причин каждого краша/фикса в [results.md](results.md#общие-проблемы-при-запуске-на-mac-cpu-и-их-решения).
