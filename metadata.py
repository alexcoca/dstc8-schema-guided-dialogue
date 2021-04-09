"""
This module contains variables that are relevant for SGD corpus analysis.
"""

from typing import Dict, List, Set

import collections
import json
import os
import logging

directory = os.path.dirname(__file__)
metadata_path = os.path.join(directory, 'metadata.json')

try:
    with open(metadata_path, 'r') as f:
        _metadata = json.load(f)  # type: dict
except (FileNotFoundError, json.decoder.JSONDecodeError):
    logging.warning("No metadata file detected for corpus or metadata file corrupt.")
    _metadata = collections.defaultdict(lambda: '')


def cast_vals_to_set(d: dict) -> dict:
    """Maps the values of a nested dictionary to a set of strings."""

    for key, value in d.items():
        if isinstance(value, dict):
            cast_vals_to_set(value)
        else:
            d[key] = set(value)
    return d


VERSION = _metadata['VERSION']
"""
The commit hash of the state of the repo which was used to generate the metadata.
"""

SPLIT_NAMES = ['train', 'dev', 'test']
"""Names of the splits in the corpus. Used by analysis functions for iterations a
cross splits or the entire corpus.
"""  # type: List[str]

SCHEMA_PATHS = {split: f"{directory}/{split}/schema.json" for split in SPLIT_NAMES}
"""Mapping with paths to the schema .json files, for each split. See metadata.SPLIT_NAMES
for key values.
"""  # type: Dict[str, str]

SERVICES_TO_FILES = _metadata['SERVICES_TO_FILES']
"""Nested mapping with files where different services can be found::
    
    {
    'split_name': {'service_name': List[str], of filenames }
    }
"""  # type: Dict[str, Dict[str, List[str]]]

ALL_INTENTS = set(_metadata['ALL_INTENTS']) if _metadata['ALL_INTENTS'] else set()
"""Set of all the intents in the corpus.
"""  # type: Set[str]

SEARCH_INTENTS = set(_metadata['SEARCH_INTENTS']) if _metadata['SEARCH_INTENTS'] else set()
"""Set of all the intents which return entities following an API call (e.g., restaurants,
calendar appointment).
"""  # type: Set[str]

INTENTS_BY_SPLIT = cast_vals_to_set(_metadata['INTENTS_BY_SPLIT']) if _metadata['INTENTS_BY_SPLIT'] else {}
"""Mapping of split names to the intents called in dialogues found in that split.
"""  # type: Dict[str, Set[str]]

TRANSACTIONAL_INTENTS = set(_metadata['TRANSACTIONAL_INTENTS']) if _metadata['TRANSACTIONAL_INTENTS'] else set()
"""Set of all the intents that are called in order to execute a transaction (e.g., booking,
adding event to a calendar)
"""  # type: Set[str]

INTENTS_TO_SERVICES = cast_vals_to_set(_metadata['INTENTS_TO_SERVICES']) if _metadata['INTENTS_TO_SERVICES'] else {}
"""A mapping with the following structure::

            {
            'split':{
                'intent_1': ['Service_1']
                `intent_2': ['Service_1', 'Service_2']
            }
"""  # type: Dict[str, Dict[str, Set[str]]]

REQUESTABLE_SLOTS = set(_metadata['REQUESTABLE_SLOTS']) if _metadata['REQUESTABLE_SLOTS'] else set()
"""Set of slots that the user requests. These represent entity attributes (e.g., address,
pets allowed).
"""  # type: Set[str]

BINARY_SLOTS = set(_metadata['BINARY_SLOTS']) if _metadata['BINARY_SLOTS'] else set()
"""Set of slots which take only ``True`` and ``False`` or ``'0'`` and ``'1'`` values. These
slots are not delexicalised in the original corpus.
"""  # type: Set[str]

_binary_by_service = _metadata['BINARY_SLOTS_BY_SERVICE']
BINARY_SLOTS_BY_SERVICE = cast_vals_to_set(_binary_by_service) if _binary_by_service else {}
"""Mapping of service names to binary slots. The union of all the values yields `BINARY_SLOTS`
"""  # type: Dict[str, Set[str]]

CATEGORICAL_SLOTS = set(_metadata['CATEGORICAL_SLOTS']) if _metadata['CATEGORICAL_SLOTS'] else set()
"""Set of slots which take a finite number of values.
"""  # type: Set[str]

_categorical_by_service = _metadata['CATEGORICAL_SLOTS_BY_SERVICE']
CATEGORICAL_SLOTS_BY_SERVICE = cast_vals_to_set(_categorical_by_service) if _categorical_by_service else {}
"""
Mapping of service names to categorical slots. 
"""  # type: Dict[str, Set[str]]

ENTITY_SLOTS = cast_vals_to_set(_metadata['ENTITY_SLOTS']) if _metadata['ENTITY_SLOTS'] else {}
"""Mapping with slots mentioned by the system when a successful call is made to a search/query intent. Format is::

    {
        'service_name': {'intent_name': {'slot_name', ...}}
    }
"""  # type: Dict[str, Dict[str, Set[str]]]

_transactional_dialogues = _metadata['TRANSACTIONAL_DIALOGUES']
TRANSACTIONAL_DIALOGUES = cast_vals_to_set(_transactional_dialogues) if _transactional_dialogues else {}
"""Mapping of split name to a set dialogue ids of dialogues that are comprised only of transactional intents.
"""  # type: Dict[str, Set[str]]

SEARCH_DIALOGUES = cast_vals_to_set(_metadata['SEARCH_DIALOGUES']) if _metadata['SEARCH_DIALOGUES'] else {}
"""Mapping of split names to dialogue ids of dialogues that are comprised only of search intents.
"""  # type: Dict[str, Set[str]]

_mixed_intent_dialogues = _metadata['MIXED_INTENT_DIALOGUES']
MIXED_INTENT_DIALOGUES = cast_vals_to_set(_mixed_intent_dialogues) if _mixed_intent_dialogues else {}
"""Mapping of split names to dialogue ids of dialogues that are comprised only of search intents.
"""  # type: Dict[str, Set[str]]

_multiple_service_dialogues = _metadata['MULTIPLE_SERVICES_DIALOGUES']
MULTIPLE_SERVICES_DIALOGUES = cast_vals_to_set(_multiple_service_dialogues) if _multiple_service_dialogues else {}
"""Mapping of split names to dialogue ids of dialogues that have multiple services.
"""  # type: Dict[str, Set[str]]
