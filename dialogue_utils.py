"""A module which contains utility function for inspecting and extracting
data from dialogues and training splits. Use in conjunction with the
information in the `metadata` module.
"""

from utils import dialogue_iterator


def has_requestables(dialogue: dict) -> bool:
    """Returns `True` if the user requests information
    from the system and false otherwise.
    """

    for turn in dialogue_iterator(dialogue, user=True, system=False):
        for frame in turn['frames']:
            if frame['state']['requested_slots']:
                return True
    return False
