"""A module which contains utility function for inspecting and extracting
data from dialogues and training splits. Use in conjunction with the
information in the `metadata` module.
"""

from typing import Dict, Set
from .data_utils import dialogue_iterator

import metadata


def has_requestables(dialogue: dict) -> bool:
    """Returns `True` if the user requests information
    from the system and false otherwise.
    """

    for turn in dialogue_iterator(dialogue, user=True, system=False):
        for frame in turn['frames']:
            if frame['state']['requested_slots']:
                return True
    return False


def get_dialogue_intents(dialogue: Dict, exclude_none: bool = True) -> Set[str]:
    """Returns the intents in a dialogue.

    Parameters
    ----------
    dialogue
        Nested dictionary containing dialogue and annotations.
    exclude_none
        If True, the `NONE` intent is not included in the intents set.

    Returns
    -------
    intents
        A set of intents contained in the dialogue.
    """

    intents = set()
    for turn in dialogue_iterator(dialogue, user=True, system=False):
        for frame in turn['frames']:
            intent = frame['state']['active_intent']
            if exclude_none:
                if intent == 'NONE':
                    continue
                else:
                    intents.add(intent)
            else:
                intents.add(intent)
    return intents


def offers_entities(dialogue: dict) -> bool:
    """Returns `True` if the system offers some entity in
    the input dialogue. This is based on whether the dialogue
    contains at least a search intent.
    """

    intents = get_dialogue_intents(dialogue, exclude_none=True)
    search_intents = set(metadata.SEARCH_INTENTS)
    return any(intent in search_intents for intent in intents)
