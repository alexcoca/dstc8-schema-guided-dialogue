from collections import defaultdict
from typing import Tuple, List, Dict, Optional, Set
from typing_extensions import Literal

import glob
import json

_SPLIT_NAMES = ['train', 'test', 'dev']
_SCHEMA_PATHS = {split: f"{split}/schema.json" for split in _SPLIT_NAMES}


def get_filename(dial_id: str) -> str:
    """Reconstruct filename from dialogue ID."""

    file_prefix = int(dial_id.split("_")[0])

    if file_prefix in range(10):
        str_file_prefix = f'00{file_prefix}'
    elif file_prefix in range(10, 100):
        str_file_prefix = f'0{file_prefix}'
    else:
        str_file_prefix = f'{file_prefix}'

    return f'dialogues_{str_file_prefix}.json'


def get_file_map(dialogue_ids: List, split: Literal['train', 'test', 'dev']) -> Dict[str, List]:
    """Returns a map where the keys are filenames and values are lists
    comprising dialogues from `dialogue_ids` that are in the same file.
    """

    file_map = defaultdict(list)

    for id in dialogue_ids:
        file_map[f"{split}/{get_filename(id)}"].append(id)

    return file_map


def file_iterator(filename: str, return_only : Optional[Set[str]] = None) -> Tuple[str, dict]:

    with open(filename, 'r') as f:
        dial_bunch = json.load(f)

    max_index = int(dial_bunch[-1]['dialogue_id'].split("_")[1]) + 1
    n_dialogues = len(dial_bunch)
    missing_dialogues = not (max_index == n_dialogues)

    if return_only:
        if not missing_dialogues:
            for dial_idx in (int(dial_id.split("_")[1]) for dial_id in return_only):
                yield filename, dial_bunch[dial_idx]
        else:
            returned = set()
            for dial in dial_bunch:
                if (found_id := dial['dialogue_id']) in return_only:
                    returned.add(found_id)
                    yield filename, dial
                    if returned == return_only:
                        break

    else:
        raise NotImplementedError


def split_iterator(split: Literal['train', 'test', 'dev'], return_only : Optional[Set[str]] = None) -> Tuple[str, Dict]:

    # return specified dialogues only
    if return_only:
        file_map = get_file_map(list(return_only), split)
        for filename, dial_ids in file_map.items():
            yield from file_iterator(filename, return_only=set(dial_ids))
    # iterate through all dialogues
    else:
        for fp in glob.glob(f"{split}/dialogues*.json"):
            with open(fp, 'r') as f:
                dial_bunch = json.load(f)
            for dial in dial_bunch:
                yield fp, dial

def corpus_iterator():

    for split in _SPLIT_NAMES:
        yield from split_iterator(split)


def dialogue_iterator(dialogue: dict, user: bool = True, system: bool = True):

    if (not user) and (not system):
        raise ValueError("At least a speaker needs to be specified!")

    filter = 'USER' if not user else 'SYSTEM' if not system else ''

    for turn in dialogue["turns"]:
        if filter and turn['speaker'] == filter:
            continue
        else:
            yield turn

def schema_iterator(split: Literal['train', 'test', 'dev']) -> dict:

    with open(_SCHEMA_PATHS[split], 'r') as f:
        schema = json.load(f)
    for service in schema:
        yield service

def dial_sort_key(dialogue_id: str) -> Tuple[int, int]:
    s1, s2 = dialogue_id.split("_")
    return int(s1), int(s2)

def alphabetical_sort_key(name: str, n_chars: int = 10) ->  str:
    return name[:n_chars]
