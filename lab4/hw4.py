"""Загрузка фотографий, манифестов и весов для двух ноутбуков."""

import json
import shutil
import ssl
import tarfile
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zipfile import ZipFile

import certifi
import pandas as pd
from filelock import FileLock

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".hw4-cache"
CACHE.mkdir(exist_ok=True)
GITHUB_REPO = "https://github.com/MakSoS1/mlt_labs"
GITHUB_REF = "main"
materials = RESULTS = MANIFESTS = None
SUMS = {
    "models/parents.zip": "f446c63a6e4f5e11612a2f1991839c44cc226f1eaaaa478aa0983b739f51d249",
    "materials.zip": "c1c6578f8e077b1912f0c77d7b27d137d0560cb3760dc3654d089cf6bb3be9f7",
    "models/required.zip": "30ced1e05e58fed5ec1ca6c05e7cd784c3f9686d6085102b47411d757b90e7f5",
    "models/winners.zip": "b90097eb48ebfacca0b7b69e102e9c8e248263de96cf2867d36ee0fadbe6e866",
}


def checksum(file):
    digest = sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, dest, expected=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and (expected is None or checksum(dest) == expected):
        return dest
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=certifi.where())
    request = Request(url, headers={"User-Agent": "HW4-notebook/1.0"})
    with NamedTemporaryFile(dir=dest.parent, delete=False) as temp:
        partial = Path(temp.name)
    try:
        with urlopen(request, timeout=120, context=context) as response, partial.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output, 1024 * 1024)
        if expected and checksum(partial) != expected:
            raise ValueError(f"Скачанный файл отличается от исходного: {dest.name}")
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)
    return dest


def repository_file(name):
    expected = SUMS[name]
    for file in (ROOT / name, CACHE / "downloads" / name):
        if file.is_file() and checksum(file) == expected:
            return file.resolve()
    url = urlparse(GITHUB_REPO.strip().rstrip("/").removesuffix(".git"))
    if url.hostname != "github.com" or len(url.path.strip("/").split("/")) != 2:
        raise ValueError(
            "Укажите в первой ячейке адрес GitHub-репозитория и повторите запуск."
        )
    slug = url.path.strip("/")
    host = (
        "media.githubusercontent.com/media"
        if name.startswith("models/")
        else "raw.githubusercontent.com"
    )
    return download(
        f"https://{host}/{slug}/{GITHUB_REF}/lab4/{name}",
        CACHE / "downloads" / name,
        expected,
    )


def unpack(name, folder):
    folder.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder) + ".lock"):
        marker = folder / ".ready"
        stamp = SUMS[name] + SUMS["materials.zip"]
        if marker.exists() and marker.read_text() == stamp:
            return folder
        with ZipFile(repository_file(name)) as archive:
            members = archive.namelist()
            if name.startswith("models/"):
                members = [key for key in members if key.endswith((".pt", ".json"))]
            archive.extractall(folder, members=members)
        marker.write_text(stamp)
        return folder


def prepare(repository="https://github.com/MakSoS1/mlt_labs", ref="main"):
    global GITHUB_REPO, GITHUB_REF, materials, RESULTS, MANIFESTS
    GITHUB_REPO, GITHUB_REF = repository, ref
    materials = unpack(
        "materials.zip", CACHE / SUMS["materials.zip"][:12] / "materials"
    )
    RESULTS = materials / "results"
    MANIFESTS = materials / "data/manifests"
    return RESULTS, MANIFESTS


def open_models(name):
    return unpack(f"models/{name}.zip", materials.parent / name)


COURSE_URL = "https://drive.usercontent.google.com/download?id=1lPLlLKhRXvtCG4SNXYflIiy09DMqqPkN&export=download&confirm=t"
COURSE_SHA = "540915915226479357a1e738bd920c3e19e5a3d8fc56c0622dffa5d682e3631c"


def extract_photos(group="course", split=None):
    if group != "course":
        raise ValueError(group)
    catalogue = pd.read_csv(MANIFESTS / "course_all.csv")
    if split is not None:
        catalogue = catalogue[catalogue.split == split]
    dest = CACHE / "data/course"
    local = ROOT / "Cats_and_Dogs.zip"
    if local.is_file():
        if checksum(local) != COURSE_SHA:
            raise ValueError("Cats_and_Dogs.zip отличается от исходного архива курса.")
        archive = local
    else:
        archive = download(
            COURSE_URL, CACHE / "downloads/Cats_and_Dogs.zip", COURSE_SHA
        )
    with ZipFile(archive) as z:
        for row in catalogue.itertuples():
            path = dest / row.relative_path
            if path.exists() and checksum(path) == row.sha256:
                continue
            data = z.read("Cats_and_Dogs/" + row.relative_path)
            if sha256(data).hexdigest() != row.sha256:
                raise ValueError(row.relative_path)
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_bytes(data)
    return dest


def download_external(group):
    files = pd.read_csv(MANIFESTS / "external_files.csv")
    rows = files[files.dataset == group]
    if rows.empty:
        raise ValueError(f"Неизвестный набор: {group}")
    dest = CACHE / "data" / group
    expected = dict(zip(rows.path, rows.sha256, strict=True))
    missing = {
        k
        for k, digest in expected.items()
        if not (dest / k).exists() or checksum(dest / k) != digest
    }
    if not missing:
        return dest

    def save(name, data):
        if sha256(data).hexdigest() != expected[name]:
            raise ValueError(f"Изображение отличается от манифеста: {name}")
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        missing.remove(name)

    if group in ("coco", "open-images"):
        table = "coco_train.csv" if group == "coco" else "open_train.csv"
        source = pd.read_csv(MANIFESTS / table).set_index("relative_path")

        def get_one(name):
            download(source.at[name, "source_url"], dest / name, expected[name])

        with ThreadPoolExecutor(max_workers=6) as workers:
            for _ in workers.map(get_one, sorted(missing)):
                pass
        return dest
    if group == "petfinder":
        audit = json.loads((MANIFESTS / "rescue_audit.json").read_text())
        base = f"https://huggingface.co/datasets/{audit['source']}/resolve/{audit['revision']}/"
        for i, digest in enumerate(audit["archive_sha256"]):
            name = f"images-{i:05d}.zip"
            archive = download(base + name, CACHE / "downloads" / name, digest)
            with ZipFile(archive) as z:
                for key in missing.intersection(z.namelist()):
                    save(key, z.read(key))
    else:
        sources = {
            "oxford": "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz",
            "stanford": "https://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
        }
        url = sources[group]
        archive = download(url, CACHE / "downloads" / group / url.rsplit("/", 1)[-1])
        with tarfile.open(archive, "r|*") as tar:
            for member in tar:
                key = member.name.removeprefix("./").removeprefix("Images/")
                if member.isfile() and key in missing:
                    with tar.extractfile(member) as stream:
                        save(key, stream.read())
    if missing:
        raise ValueError(f"{group}: в источнике не найдены {len(missing)} файлов")
    return dest


def demo_photo():
    return download(
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        CACHE / "demo/dog.jpg",
        "f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a",
    )
