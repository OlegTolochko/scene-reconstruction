from evaluation.datasets.base import Dataset, View
from evaluation.datasets.scannetpp import ScanNetPP

DATASETS = {"scannetpp": ScanNetPP}


def get(name: str) -> Dataset:
    return DATASETS[name]()


__all__ = ["DATASETS", "Dataset", "View", "get"]
