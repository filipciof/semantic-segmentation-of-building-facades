from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .dataloader import (
    CMP_facade_DB_base,
    CMP_facade_DB_extended,
    convert_mask,
    load_images_and_masks,
)


def get_train_test_val(
    dirs: list[Path] | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    include_original_masks: bool = True,
) -> dict[str, dict[str, list[np.ndarray]]]:
    """Wczytuje podział CMP na listy obrazów i masek w natywnych rozmiarach.

    ``masks`` zawiera maski po mapowaniu na osiem klas. Opcjonalne
    ``og_masks`` zawiera maski z etykietami źródłowymi.
    """
    if dirs is None:
        dirs = [CMP_facade_DB_base, CMP_facade_DB_extended]

    images, masks = load_images_and_masks(dirs)

    train_val_idx, test_idx = train_test_split(
        np.arange(len(images)),
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
    )

    result = {}
    for name, indices in (("train", train_idx), ("test", test_idx), ("val", val_idx)):
        original_masks = [masks[index] for index in indices]
        result[name] = {
            "images": [images[index] for index in indices],
            "masks": [convert_mask(mask) for mask in original_masks],
        }
        if include_original_masks:
            result[name]["og_masks"] = original_masks

    return result
