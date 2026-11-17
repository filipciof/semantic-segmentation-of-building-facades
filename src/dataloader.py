import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split

# Wydawać by się mogło, że te liczby po prawej to wartości
# pikseli w masce, które odpowiadają etykiecie, ale nic
# bardziej mylnego. Te po lewej są prawidłowe. Może te
# po prawej są w tych XML-ach???

BACKGROUND = 1
FACADE = 2
WINDOW = 3
DOOR = 4
CORNICE = 5
SILL = 6
BALCONY = 7
BLIND = 8
DECO = 9
MOLDING = 10
PILLAR = 11
SHOP = 12

# Wartości pikseli w maskach po konwersji.

BACKGROUND_NEW = 1
FACADE_NEW = 2
WINDOW_NEW = 3
DOOR_NEW = 4
SILL_NEW = 5
BALCONY_NEW = 6
SHOP_NEW = 7
PILLAR_NEW = 8

CMP_facade_DB_base = Path("datasets/CMP_facade_DB_base/base")
CMP_facade_DB_extended = Path("datasets/CMP_facade_DB_extended/extended")


def load_images_and_masks(dirs: list[Path]) -> tuple[list, list]:
    """
    Zakładamy, że oryginalne zdjęcia są .jpg a maski .png
    z tą samą nazwą.
    """

    def load_from_one_dir(dir: Path):
        imgs = []
        masks = []

        for item in sorted(dir.iterdir()):
            if item.is_file() and item.suffix.lower() == ".jpg":
                mask_path = item.with_suffix(".png")

                with Image.open(item) as img:
                    imgs.append(np.array(img))

                with Image.open(mask_path) as mask:
                    masks.append(np.array(mask))

        return imgs, masks

    img_lst, mask_lst = [], []

    for dir in dirs:
        imgs, masks = load_from_one_dir(dir)
        img_lst.extend(imgs)
        mask_lst.extend(masks)

    return img_lst, mask_lst


def resize_images_and_masks(
    imgs: list, masks: list, img_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oryginalne: BILINEAR
    Maski: NEAREST
    """

    np_imgs = np.stack(
        [
            np.array(
                Image.fromarray(img).resize(
                    (img_size, img_size),
                    resample=Image.Resampling.BILINEAR,
                )
            )
            for img in imgs
        ]
    )

    np_masks = np.stack(
        [
            np.array(
                Image.fromarray(mask).resize(
                    (img_size, img_size),
                    resample=Image.Resampling.NEAREST,
                )
            )
            for mask in masks
        ]
    )

    return np_imgs, np_masks


def get_label_name(label: int) -> str:
    """
    Zwraca nazwę klasy na podstawie etykiety (1-8) po konwersji.
    """

    labels = get_labels()
    return labels[label - 1] if 1 <= label <= len(labels) else "null"


def get_labels() -> list[str]:
    return [
        "background",
        "facade",
        "window",
        "door",
        "sill",
        "balcony",
        "shop",
        "pillar",
    ]


def get_old_labels() -> list[str]:
    return [
        "background",
        "facade",
        "window",
        "door",
        "cornice",
        "sill",
        "balcony",
        "blind",
        "deco",
        "molding",
        "pillar",
        "shop",
    ]


def convert_mask(mask: np.ndarray) -> np.ndarray:
    """
    Mapuje oryginalne etykiety (1-12) na nowe etykiety (1-8).
    Działa na jednej masce (H, W) lub kilku maskach (N, H, W).
    Przyjmuje maski z oryginalnymi etykietami, także po zmianie rozmiaru.
    """

    convert_dict = {
        BACKGROUND: BACKGROUND_NEW,
        FACADE: FACADE_NEW,
        WINDOW: WINDOW_NEW,
        DOOR: DOOR_NEW,
        CORNICE: FACADE_NEW,
        SILL: SILL_NEW,
        BALCONY: BALCONY_NEW,
        BLIND: WINDOW_NEW,
        DECO: FACADE_NEW,
        MOLDING: FACADE_NEW,
        PILLAR: PILLAR_NEW,
        SHOP: SHOP_NEW,
    }

    converted_mask = np.copy(mask)

    for old, new in convert_dict.items():
        converted_mask[mask == old] = new

    return converted_mask


def get_train_test_val(
    dirs=[CMP_facade_DB_base, CMP_facade_DB_extended],
    img_size=256,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
    include_original_masks=True,
) -> dict:
    """
    Główna funkcja do wczytywania danych z CMP_facade_DB.
    Zdjęcia i maski mają zmieniony rozmiar. Maski w "masks"
    mają nowe etykiety (1-8). Jeśli include_original_masks=True,
    "og_masks" zawiera maski z oryginalnymi etykietami (1-12).

    Zwraca dict:
    {
        "train" : {
            "images": images,
            "masks": masks,
            "og_masks": og_masks
        },
        "test": {
            "images": images,
            "masks": masks,
            "og_masks": og_masks
        },
        "val": {
            "images": images,
            "masks": masks,
            "og_masks": og_masks
        },
    }
    """

    imgs, masks = load_images_and_masks(dirs)
    imgs_resized, masks_resized = resize_images_and_masks(
        imgs, masks, img_size
    )
    masks_converted = convert_mask(masks_resized)

    train_val_idx, test_idx = train_test_split(
        np.arange(len(imgs)),
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
    )

    result = {}
    for name, indices in [
        ("train", train_idx),
        ("test", test_idx),
        ("val", val_idx),
    ]:
        result[name] = {
            "images": imgs_resized[indices],
            "masks": masks_converted[indices],
        }

        if include_original_masks:
            result[name]["og_masks"] = masks_resized[indices]

    return result
    