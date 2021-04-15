from collections import defaultdict
from dialogue_utils import get_dialogue_intents
from typing import Dict, Set, Tuple, List, Optional, Callable
from typing_extensions import Literal
from utils import (
    get_filenames,
    corpus_iterator,
    schema_iterator,
    file_iterator,
    split_iterator,
    dialogue_iterator,
    dial_sort_key,
    alphabetical_sort_key,
    dial_files_sort_key,
)


import collections
import json
import os
import subprocess


_SPLIT_NAMES = ['train', 'dev', 'test']  # type: List[Literal['train'], Literal['dev'], Literal['test'] ]


def cast_vals_to_sorted_list(d: dict, sort_by: Optional[Callable] = None) -> dict:
    """Casts the values of a nested dict to sorted lists.

    Parameters
    ----------
    d
    sort_by:
        A callable to be used as sorting key.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            cast_vals_to_sorted_list(value)
        else:
            d[key] = sorted(list(value), key=sort_by)

    return d


def get_commit_hash():
    """Returns the commit hash for the current HEAD."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()


def get_schema_intents(split: Literal['train', 'test', 'dev']) -> Dict[str, Set[str]]:
    """Returns a mapping of intent type (transactional/search) to a set of intents for
    the required split.

    Parameters
    ----------
    split
        The split from which the intents are retrieved.

    Returns
    -------
    intents
        A mapping from intent type to a set of intent names of that type found in `split`.
    """

    intents = {'transactional': set(), 'search': set()}
    for service in schema_iterator(split):
        for intent in service['intents']:
            if intent['is_transactional']:
                intents['transactional'].add(intent['name'])
            else:
                intents['search'].add(intent['name'])

    return intents


def get_intents_by_split() -> Dict[str, List[str]]:
    """Returns a mapping from split names to the intents in that split.
    """

    intents_by_split = {}
    for split in _SPLIT_NAMES:
        intents_by_split[split] = set().union(*get_schema_intents(split).values())

    return cast_vals_to_sorted_list(intents_by_split, sort_by=alphabetical_sort_key)


def get_intents_by_type() -> Dict[str, List[str]]:
    """Returns a mapping containing two keys:
    
            - ``'transactional'``: the intent represented implies \\
            a transaction (e.g.,BookRestaurant)
            
            - ``'search'``: the intent represented is a query (e.g., FindMusic)
    
    Returns
    -------
    all_intents
        A dictionary containing sorted lists of transactional and search intents as value sets.
    """  # noqa
    transactional_intents = set()
    search_intents = set()
    for split in _SPLIT_NAMES:
        schema_intents = get_schema_intents(split)
        transactional_intents.update(
            schema_intents['transactional'])
        search_intents.update(
            schema_intents['search']
        )

    all_intents = {
        'transactional': sorted(list(transactional_intents), key=alphabetical_sort_key),
        'search': sorted(list(search_intents), key=alphabetical_sort_key)
    }

    return all_intents


def _get_requestables(dialogue: dict) -> Set[str]:
    """Get requestable slots in a given dialogue.

    Parameters
    ----------
    dialogue
        A nested dictionary representation of the dialogue in SGD format.

    Returns
    -------
    requestables
        Set of slots which appear in the ``['state']['requested_slots']`` of all user frames.
    """

    requestables = set()
    for turn in dialogue_iterator(dialogue, user=True, system=False):
        for frame in turn['frames']:
            if reqs := frame['state']['requested_slots']:
                requestables.update(reqs)
    return requestables


def get_requestable_slots() -> List[str]:
    """Return a list of all the slots requested by the user across the entire corpus.
    These slots are specified under the ``['state']['requestables']``

    Returns
    -------
    all_requestables
        A sorted list containing the slot names of the requestable slots across the entire corpus.
    """
    all_requestables = set()
    for _, dialogue in corpus_iterator():
        all_requestables.update(_get_requestables(dialogue))

    return sorted(list(all_requestables), key=alphabetical_sort_key)


def _find_service_binary_slots(service: dict) -> dict:
    """Find categorical slots with binary values (i.e., True/False, 0/1) for a given schema.

    Parameters
    ----------
    service
        A nested dictionary representation of the schema.

    Returns
    -------
    binary_slots
        A dictionary mapping the names of the binary slots to their values.

    # TODO: LOOK INTO 0/1 VAL CASES AND SEE IF THESE ARE ACTUALLY DELEX (transfers slot is, for example)
    """

    binary_slots = collections.defaultdict(set)
    service_name = service['service_name']
    for slot in service['slots']:
        if slot['is_categorical']:
            if len(values := slot['possible_values']) == 2:
                condition_1 = 'True' in values
                condition_2 = values[0].isdigit() and int(values[0]) <= 1
                if condition_1 or condition_2:
                    binary_slots[service_name].add(slot['name'])

    # add an empty set for the services that do not have binary slots
    # to ensure consistency between CATEGORICAL_SLOTS_BY_SERVICE and BINARY_SLOTS_BY SERVICE
    if not binary_slots:
        binary_slots[service_name] = set()

    return binary_slots


def get_binary_slots() -> Tuple[List[str], Dict[str, List[str]]]:
    """Get names of slots with binary values. For these slots, there is no
    delexicalisation annotation.

    Returns
    -------
    service_binary_slots
        Mapping from services to the binary slots contained in their schemas.
    binary_slots
        Set of binary slots
    """
    service_binary_slots = {}
    for split in _SPLIT_NAMES:
        for service in schema_iterator(split):
            binary_slots = _find_service_binary_slots(service)
            service_binary_slots.update(binary_slots)

    # cast and sort output for writing to .json
    binary_slots = list(set.union(*list(service_binary_slots.values())))
    binary_slots.sort(key=alphabetical_sort_key)

    return binary_slots, cast_vals_to_sorted_list(service_binary_slots)


def _get_service_categorical_slots(service: dict, binary_slots: List[str]) -> Dict[str, List[str]]:
    """Returns  the categorical slots in `service`. Binary slots with values `True`/`False`
    are not considered as categorical slots.

    Returns
    -------
    cat_slots
        A mapping from slot names to their values.
    """

    cat_slots = collections.defaultdict(list)
    for slot_dict in service['slots']:
        slot_name = slot_dict['name']
        values = slot_dict['possible_values']
        if slot_dict['is_categorical'] and slot_name not in binary_slots:
            cat_slots[slot_name].extend(values)

    return cat_slots


def get_categorical_slots(binary_slots_by_service: Dict[str, List[str]]) -> \
        Tuple[List[str], Dict[str, Dict[str, List[str]]]]:
    """Find all categorical slots in the corpus.

    Parameters
    ----------
    binary_slots_by_service
        A mapping from service names to the names of slots that take only ``True``/``False`` and
        ``0`` and ``1`` values.

    Returns
    -------
    all_cat_slots
        A set of all categorical slots in the corpus.
    cat_slots_by service
        A mapping with the structure::

            {
            'service_name': {'slot_name': List[str], values of slot indicated in key}
            }

        where the slot names are categorical slots (i.e., take a finite number of values).
    """

    cat_slots_by_service = {}
    all_cat_slots = set()
    for split in _SPLIT_NAMES:
        for service in schema_iterator(split):
            service_name = service['service_name']
            binary_slots = []
            if service_name in binary_slots_by_service:
                binary_slots = binary_slots_by_service[service_name]
            service_cat_slots = _get_service_categorical_slots(service, binary_slots)
            all_cat_slots.update(list(service_cat_slots.keys()))
            if service_name in cat_slots_by_service:
                for slot_name, values in cat_slots_by_service[service_name].items():
                    assert slot_name in service_cat_slots
                    assert values == service_cat_slots[slot_name]
            else:
                cat_slots_by_service[service_name] = service_cat_slots

    return list(all_cat_slots), cat_slots_by_service


def filter_by_intent_type(split: Literal['train', 'test', 'dev'],
                          transactional: bool = True,
                          search: bool = False) -> Dict[str, set]:
    """Filters dialogues so only specific types of intents are contained.

    Parameters
    ----------
    split
        Split to be filtered
    transactional
        If `True`, dialogue IDs returned will contain transactional intents.
    search
        If `True`, dialogue IDs returned will contain search intents.

    Returns
    -------
    A mapping from filenames to a set of dialogue IDs where the dialogues that
    contain the types of intents indicated by the kwargs.

    Raises
    ------
    ValueError
        If both kwargs are False (one of the two types of intents necessarily appears in a dialogue).
    """

    if not transactional and not search:
        raise ValueError("At least one intent type must be specified")

    all_intents = get_intents_by_type()
    transactional_intents = all_intents['transactional']
    search_intents = all_intents['search']

    dialogue_ids = defaultdict(set)
    for fp, dial in split_iterator(split):
        dialogue_id = dial['dialogue_id']
        intents = get_dialogue_intents(dial)
        if transactional and not search:
            if intents.issubset(transactional_intents):
                dialogue_ids[fp].add(dialogue_id)
        elif search and not transactional:
            if intents.issubset(search_intents):
                dialogue_ids[fp].add(dialogue_id)
        else:
            transctional_subset = intents.issubset(transactional_intents)
            search_subset = intents.issubset(search_intents)

            if not transctional_subset and not search_subset:
                dialogue_ids[fp].add(dialogue_id)

    return dialogue_ids


def _get_entity_slots(split: Literal['train', 'test', 'dev']) -> Dict[str, Dict[str, Set[str]]]:
    """Find the slots that are always specified by the system when a call to a "search" intent
    is made (referred to as "entity" slots).

    Parameters
    ----------
    split
        The data split for which the entity slots are to be determined

    Returns
    -------
    entity_slots_map
        A mapping of the form::

            {
            'service_name':
                         {'intent_1': {'slot_name',...}
            ...
            }

    """
    # select only dialogues where the system speaks about an entity following a search call
    filtered_dialogues = filter_by_intent_type(split, transactional=True, search=True)
    search_only = filter_by_intent_type(split, transactional=False, search=True)
    for file_id, dial_ids in search_only.items():
        if file_id in filtered_dialogues:
            filtered_dialogues[file_id].update(dial_ids)
        else:
            filtered_dialogues[file_id] = dial_ids

    # iterate through selected dialogues to find slots that are always specified after a
    # call to a given intent ("entity slot").
    entity_slots_map = defaultdict(lambda: collections.defaultdict(set))
    search_intents = set(get_intents_by_type()['search'])
    for file in filtered_dialogues.keys():
        for fp, dial in file_iterator(file, return_only=filtered_dialogues[file]):
            for turn in dialogue_iterator(dial, user=False, system=True):
                if 'service_call' in (frame := turn['frames'][0]):
                    service = frame['service']
                    intent = frame['service_call']['method']
                    service_results = frame['service_results']
                    if intent in search_intents and service_results:
                        mentioned_slots = {entry['slot'] for entry in frame['slots']}
                        if intent in entity_slots_map[service]:
                            entity_slots_map[service][intent] = \
                                entity_slots_map[service][intent].intersection(mentioned_slots)
                        else:
                            entity_slots_map[service][intent] = mentioned_slots
    return entity_slots_map


def get_entity_slots_map() -> Dict[str, Dict[str, Set[str]]]:
    """Returns a nested map from service name to intent name to the slots that are always
    specified by the system following a _successful_ call to a search/query intent.

    Returns
    -------
    entity_slots
        A mapping of the form::

            {
            'service_name':
                         {'intent_1': {'slot_name',...}
            ...
            }

        which contains the entity slots for all the services and intents in the corpus.
    """

    # TODO: IMPROVE THIS IN SPARE TIME, SHOULD BE LESS VERBOSE
    entity_slots = {}
    for split in _SPLIT_NAMES:
        this_split_slots = _get_entity_slots(split)
        if not entity_slots:
            entity_slots = this_split_slots
        else:
            for service in this_split_slots:
                if service not in entity_slots:
                    entity_slots[service] = this_split_slots[service]
                else:
                    for intent in this_split_slots[service]:
                        if intent not in this_split_slots[service]:
                            entity_slots[service][intent] = this_split_slots[service][intent]
                        else:
                            entity_slots[service][intent].update(this_split_slots[service][intent])

    # cast and covert to list for writing to .json
    for service in entity_slots:
        for intent in entity_slots[service]:
            entity_slots[service][intent] = list(entity_slots[service][intent])
            entity_slots[service][intent].sort(key=alphabetical_sort_key)

    return entity_slots


def _map_intents_to_services() -> Dict[str, Dict[str, List[str]]]:
    """Create a map of intents to services. The same intent (e.g., `FindRestaurant`) can
    be part of multiple service APIs (e.g., `Restaurant_1` and `Restaurant_2`.

    Returns
    -------
    intents_to_services
        A mapping of the form::

            {
            'split':{
                'intent_1': ['Service_1']
                `intent_2': ['Service_1', 'Service_2']
            }
    """
    intents_to_services = defaultdict(lambda: defaultdict(list))
    for split in _SPLIT_NAMES:
        for service in schema_iterator(split):
            for intent in service["intents"]:
                intent_name = intent['name']
                intents_to_services[split][intent_name].append(
                    service['service_name'])

    return intents_to_services


def get_dialogues_by_type(intents_mapping: dict) -> Dict[str, Dict[str, List[str]]]:
    """Returns a mapping from dialogue type (transactional intent only, search intent only,
    mixed intent) to dialogue IDs.

    Returns
    -------
    dialogues_by_type:
        A mapping of the form::

            {
            'transactional: ['dialogue_id',...],
            'search': ['dialogue_id',...],
            'mixed_intent': ['dialogue_id',...]
            }
    """

    dialogues_by_type = collections.defaultdict(lambda: collections.defaultdict(list))

    for split in _SPLIT_NAMES:
        for _, dial in split_iterator(split):
            dialogue_id = dial['dialogue_id']
            transactional, search = False, False
            for turn in dialogue_iterator(dial, user=True, system=False):
                for frame in turn['frames']:
                    active_intent = frame['state']['active_intent']
                    if active_intent != 'NONE':
                        if active_intent in intents_mapping['transactional']:
                            transactional = True
                        else:
                            search = True
                if transactional and search:
                    break
            if transactional and not search:
                dialogues_by_type[split]['transactional'].append(dialogue_id)
            elif search and not transactional:
                dialogues_by_type[split]['search'].append(dialogue_id)
            else:
                dialogues_by_type[split]['mixed_intent'].append(dialogue_id)

        for intent_type in dialogues_by_type:
            dialogues_by_type[split][intent_type].sort(key=dial_sort_key)

    return dialogues_by_type


def get_services(split: Literal['train', 'test', 'dev']) -> Set[str]:
    """Returns a set of services invoked in a dataset split.

    Parameters
    ----------
    split
        The split for which services are to be returned.

    Returns
    -------
        Set of services invoked in `split`.
    """
    return {service['service_name'] for service in schema_iterator(split)}


def get_file_services(filename: str) -> Set[str]:
    """Returns a list of services in a given dialogue file. The file must
    be prefixed by the split name (e.g., `train/dialogues_001.json`).
    """

    return set().union(*(dial['services'] for _, dial in file_iterator(filename)))


def get_service_to_file_map() -> Dict[str, Dict[str, List[str]]]:
    """Returns a mapping of service to files.

    split_to_files
        A mapping with the structure::

            {
            'split_name': {'service_name': List[str], where each element is a filename which appears in `split_name`}
            }
    """

    split_to_files = {split: get_filenames(split) for split in _SPLIT_NAMES}
    splits_to_services_files = collections.defaultdict(lambda: collections.defaultdict(set))
    for split in _SPLIT_NAMES:
        services = get_services(split)
        for service in services:
            splits_to_services_files[split][service] = set(split_to_files[split])
        for file in split_to_files[split]:
            this_file_services = get_file_services(file)
            missing_services = services - set(this_file_services)
            for entry in missing_services:
                splits_to_services_files[split][entry].remove(file)

    return cast_vals_to_sorted_list(splits_to_services_files, sort_by=dial_files_sort_key)


def get_multiple_services_dialogues() -> Dict[str, List[str]]:
    """Find all the dialogues in the corpus which contain multiple services. Can
    be used in combination with `utils.split_iterator()` to iterate only through
    the multi-service dialogues.

    Returns
    -------
    A mapping with the structure::

        {
        'split_name': List[str], where each element is a dialogue ID
        }
    """
    multi_service = defaultdict(list)
    for fp, dialogue in corpus_iterator():
        *_, split = os.path.dirname(fp).split("/")
        multi_service[split].append(dialogue['dialogue_id'])

    return multi_service


if __name__ == '__main__':

    # find transactional and search/query intents in the corpus
    _intents_by_type = get_intents_by_type()
    # map the dialogues of each split to their type (transactional/search/mixed_intent)
    _dialogues_by_type = get_dialogues_by_type(_intents_by_type)
    transactional_dialogues = {
        split: _dialogues_by_type[split]['transactional'] for split in _dialogues_by_type}
    search_dialogues = {
        split: _dialogues_by_type[split]['search'] for split in _dialogues_by_type
    }
    mixed_intent_dialogues = {
        split: _dialogues_by_type[split]['mixed_intent'] for split in _dialogues_by_type
    }
    # find binary slots and their breakdown by service
    binary_slots, binary_slots_by_service = get_binary_slots()
    # find categorical slots for each intent in each service
    categorical_slots, categorical_slots_by_service = get_categorical_slots(binary_slots_by_service)
    # find entity slots and perform conversions to list
    entity_slots = get_entity_slots_map()

    metadata = {
        'VERSION': get_commit_hash(),
        'SERVICES_TO_FILES': get_service_to_file_map(),
        'ALL_INTENTS': sorted(
            _intents_by_type['search'] + _intents_by_type['transactional'], key=alphabetical_sort_key),
        'SEARCH_INTENTS': _intents_by_type['search'],
        'TRANSACTIONAL_INTENTS': _intents_by_type['transactional'],
        'INTENTS_TO_SERVICES': _map_intents_to_services(),
        'INTENTS_BY_SPLIT': get_intents_by_split(),
        'REQUESTABLE_SLOTS': get_requestable_slots(),
        'BINARY_SLOTS': binary_slots,
        'BINARY_SLOTS_BY_SERVICE': binary_slots_by_service,
        'CATEGORICAL_SLOTS': categorical_slots,
        'CATEGORICAL_SLOTS_BY_SERVICE': categorical_slots_by_service,
        'ENTITY_SLOTS_BY_SERVICE': entity_slots,
        'SPLIT_NAMES': _SPLIT_NAMES,
        'TRANSACTIONAL_DIALOGUES': transactional_dialogues,
        'SEARCH_DIALOGUES': search_dialogues,
        'MIXED_INTENT_DIALOGUES': mixed_intent_dialogues,
        'MULTIPLE_SERVICES_DIALOGUES': get_multiple_services_dialogues(),
    }

    with open('metadata.json', 'w') as f:
        json.dump(metadata, f, sort_keys=True, indent=4)
    print("")
