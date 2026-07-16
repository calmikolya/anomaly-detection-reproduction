#!/usr/bin/env python3
"""
Простая обёртка над уже готовым и рабочим кодом пяти моделей.

Ничего не меняет в самих моделях (patchcore-inspection/, padim_run/, STFPM/,
SimpleNet/, DRAEM/) — просто вызывает те же самые команды, которые уже
использовались для получения результатов, и вытаскивает метрики из тех же
файлов/логов, куда модели сами их пишут.

Использование:
    python run.py --model patchcore --category bottle
    python run.py --model stfpm --category bottle --epochs 100
    python run.py --model draem --category bottle --epochs 8 --action test
    python run.py --model patchcore --category bottle --dry-run   # только показать команду
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MVTEC_PATH = os.path.expanduser("~/anomaly_local_data/mvtec")
DTD_IMAGES_PATH = os.path.expanduser("~/anomaly_local_data/dtd/images")

PATCHCORE_PY = "/opt/anaconda3/envs/patchcore/bin/python"
ANOMALIB_PY = "/opt/anaconda3/envs/anomalib/bin/python"

# Тот же порядок категорий, что и в оригинальном train_DRAEM.py (obj_batch) —
# нужен, чтобы перевести --category в --obj_id, который ждёт скрипт DRAEM.
DRAEM_OBJ_LIST = [
    "capsule", "bottle", "carpet", "leather", "pill", "transistor", "tile",
    "cable", "zipper", "toothbrush", "metal_nut", "hazelnut", "screw", "grid", "wood",
]

BASE_ENV = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")


def run_cmd(cmd, cwd, env, dry_run):
    printable = " ".join(cmd)
    print(f"[run.py] cwd={cwd}")
    print(f"[run.py] {printable}")
    if dry_run:
        return None
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    print(result.stdout[-4000:])
    if result.returncode != 0:
        print(result.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"Команда завершилась с ошибкой (код {result.returncode})")
    return result.stdout


def read_auroc_csv(csv_path, category):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r["Row Names"] == f"mvtec_{category}")
    return {
        "image_auroc": float(row["instance_auroc"]),
        "pixel_auroc": float(row["full_pixel_auroc"]),
        "pro_auroc": float(row["anomaly_pixel_auroc"]),
    }


def run_patchcore(category, dry_run):
    cwd = os.path.join(ROOT, "patchcore-inspection")
    log_group = f"IM224_WR50_{category}"
    results_dir = f"results_{category}"
    cmd = [
        PATCHCORE_PY, "bin/run_patchcore.py", "--seed", "0", "--save_patchcore_model",
        "--log_group", log_group, "--log_project", "MVTecAD_Results", results_dir,
        "patch_core", "-b", "wideresnet50", "-le", "layer2", "-le", "layer3",
        "--pretrain_embed_dimension", "1024", "--target_embed_dimension", "1024",
        "--anomaly_scorer_num_nn", "1", "--patchsize", "3",
        "sampler", "-p", "0.1", "approx_greedy_coreset",
        "dataset", "--num_workers", "0", "--resize", "256", "--imagesize", "224",
        "-d", category, "mvtec", MVTEC_PATH,
    ]
    env = dict(BASE_ENV, PYTHONPATH="src", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    run_cmd(cmd, cwd, env, dry_run)
    if dry_run:
        return None
    csv_path = os.path.join(cwd, results_dir, "MVTecAD_Results", log_group, "results.csv")
    return read_auroc_csv(csv_path, category)


def run_padim(category, dry_run):
    cwd = os.path.join(ROOT, "padim_run")
    if category != "bottle":
        raise ValueError(
            "run_padim.py сейчас поддерживает только category=bottle (категория и путь "
            "к датасету зашиты внутри скрипта). Чтобы прогнать другую категорию, нужно "
            "поправить эти две строки в padim_run/run_padim.py вручную."
        )
    cmd = [ANOMALIB_PY, "run_padim.py"]
    run_cmd(cmd, cwd, dict(BASE_ENV), dry_run)
    if dry_run:
        return None
    with open(os.path.join(cwd, "metrics.json")) as f:
        metrics = json.load(f)
    m = metrics[0] if isinstance(metrics, list) else metrics
    return {
        "image_auroc": m.get("image_AUROC"),
        "pixel_auroc": m.get("pixel_AUROC"),
    }


def run_stfpm(category, epochs, action, dry_run):
    cwd = os.path.join(ROOT, "STFPM")
    env = dict(BASE_ENV)
    if action in ("train", "all"):
        train_cmd = [
            PATCHCORE_PY, "main.py", "train",
            "--mvtec-ad", MVTEC_PATH, "--category", category,
            "--epochs", str(epochs), "--model-save-path", "snapshots",
        ]
        run_cmd(train_cmd, cwd, env, dry_run)
    if action in ("test", "all"):
        test_cmd = [
            PATCHCORE_PY, "main.py", "test",
            "--mvtec-ad", MVTEC_PATH, "--category", category,
            "--checkpoint", f"snapshots/{category}/best.pth.tar",
        ]
        stdout = run_cmd(test_cmd, cwd, env, dry_run)
        if dry_run:
            return None
        match = re.search(
            r"Pixel-AUC:\s*([\d.]+)\s*Image-AUC:\s*([\d.]+)\s*PRO:\s*([\d.]+)", stdout
        )
        if not match:
            raise RuntimeError("Не нашла строку с метриками в выводе STFPM test")
        pixel_auc, image_auc, pro = match.groups()
        return {
            "image_auroc": float(image_auc),
            "pixel_auroc": float(pixel_auc),
            "pro_auroc": float(pro),
        }
    return None


def run_simplenet(category, epochs, dry_run):
    cwd = os.path.join(ROOT, "SimpleNet")
    log_group = f"simplenet_{category}"
    results_dir = f"results_{category}"
    cmd = [
        PATCHCORE_PY, "main.py", "--seed", "0",
        "--log_group", log_group, "--log_project", "MVTecAD_Results",
        "--results_path", results_dir, "--run_name", "run",
        "net", "-b", "wideresnet50", "-le", "layer2", "-le", "layer3",
        "--pretrain_embed_dimension", "1536", "--target_embed_dimension", "1536",
        "--patchsize", "3", "--meta_epochs", str(epochs), "--embedding_size", "256",
        "--gan_epochs", "4", "--noise_std", "0.015", "--dsc_hidden", "1024",
        "--dsc_layers", "2", "--dsc_margin", ".5", "--pre_proj", "1",
        "dataset", "--num_workers", "0", "--batch_size", "8",
        "--resize", "256", "--imagesize", "224", "-d", category, "mvtec", MVTEC_PATH,
    ]
    run_cmd(cmd, cwd, dict(BASE_ENV), dry_run)
    if dry_run:
        return None
    csv_path = os.path.join(cwd, results_dir, "MVTecAD_Results", log_group, "run", "results.csv")
    return read_auroc_csv(csv_path, category)


def run_draem(category, epochs, action, dry_run):
    cwd = os.path.join(ROOT, "DRAEM")
    if category not in DRAEM_OBJ_LIST:
        raise ValueError(f"Неизвестная категория для DRAEM: {category}")
    obj_id = DRAEM_OBJ_LIST.index(category)
    run_name = f"DRAEM_test_0.0001_{epochs}_bs8_{category}_"

    if action in ("train", "all"):
        train_cmd = [
            PATCHCORE_PY, "train_DRAEM.py", "--obj_id", str(obj_id), "--bs", "8",
            "--lr", "0.0001", "--epochs", str(epochs),
            "--data_path", MVTEC_PATH + "/", "--anomaly_source_path", DTD_IMAGES_PATH,
            "--checkpoint_path", "checkpoints", "--log_path", "logs",
        ]
        run_cmd(train_cmd, cwd, dict(BASE_ENV), dry_run)

    if action not in ("test", "all"):
        return None
    if dry_run:
        print(f"[run.py] (test) в отдельном процессе Python вызвала бы "
              f"test(['{category}'], '{MVTEC_PATH}/', 'checkpoints', "
              f"'DRAEM_test_0.0001_{epochs}_bs8')")
        return None

    # test_DRAEM.test() — обычная питоновская функция, не CLI, поэтому вызываем
    # её напрямую (в дочернем процессе через -c, чтобы не тянуть в run.py все
    # тяжёлые зависимости DRAEM/torch без необходимости).
    code = (
        "import sys, json, io, contextlib; sys.path.insert(0, '.'); "
        "from test_DRAEM import test; "
        f"buf = io.StringIO()\n"
        "with contextlib.redirect_stdout(buf):\n"
        f"    test(['{category}'], '{MVTEC_PATH}/', 'checkpoints', "
        f"'DRAEM_test_0.0001_{epochs}_bs8')\n"
        "print('===RUN_PY_CAPTURE_START===')\n"
        "print(buf.getvalue())\n"
        "print('===RUN_PY_CAPTURE_END===')\n"
    )
    test_cmd = [PATCHCORE_PY, "-c", code]
    stdout = run_cmd(test_cmd, cwd, dict(BASE_ENV), dry_run=False)
    image_match = re.search(r"AUC Image:\s*([\d.]+)", stdout)
    pixel_match = re.search(r"AUC Pixel:\s*([\d.]+)", stdout)
    if not (image_match and pixel_match):
        raise RuntimeError("Не нашла метрики в выводе DRAEM test")
    return {
        "image_auroc": float(image_match.group(1)),
        "pixel_auroc": float(pixel_match.group(1)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                         choices=["patchcore", "padim", "stfpm", "simplenet", "draem"])
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--epochs", type=int, default=None,
                         help="для stfpm (по умолч. 100), simplenet (по умолч. 10), draem (по умолч. 8)")
    parser.add_argument("--action", default="all", choices=["train", "test", "all"],
                         help="для stfpm/draem — можно прогнать только train или только test")
    parser.add_argument("--dry-run", action="store_true",
                         help="только показать команду, которая будет вызвана, ничего не запускать")
    args = parser.parse_args()

    if args.model == "patchcore":
        metrics = run_patchcore(args.category, args.dry_run)
    elif args.model == "padim":
        metrics = run_padim(args.category, args.dry_run)
    elif args.model == "stfpm":
        metrics = run_stfpm(args.category, args.epochs or 100, args.action, args.dry_run)
    elif args.model == "simplenet":
        metrics = run_simplenet(args.category, args.epochs or 10, args.dry_run)
    elif args.model == "draem":
        metrics = run_draem(args.category, args.epochs or 8, args.action, args.dry_run)

    if metrics:
        print("\n=== МЕТРИКИ ===")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
