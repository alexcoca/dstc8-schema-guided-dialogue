from typing import Dict, List

import json

import numpy as np

np.random.seed(0)



def get_utterances(dialogue: Dict) -> List[str]:
    """
    Retrieves all utterances from a dialogue.

    Parameters
    ----------
    dialogue
        See `get_dialogue_outline` for structure.

    Returns
    -------
        Utterances in the input dialogue.
    """
    return [turn['utterance'] for turn in dialogue['turns']]


def print_dialogue(dialogue: Dict):
    """
    Parameters
    ----------
    dialogue
        See `get_dialogue_outline` for structure.
    """
    for i, turn in enumerate(get_utterances(dialogue)):
        print(f"{i + 1}: {turn}")


def get_actions(turn: Dict) -> List[str]:
    """
    Retrieve actions from a given dialogue turn. An action is a parametrised dialogue act
    (e.g., INFORM(price=cheap)).

    Parameters
    ----------
    turn
        Contains turn and annotations, with the structure::

            {
            'frames': [
                    {
                        'actions': Dict,
                        'service': str,
                        'slots': List[Dict], can be empty if no slots are mentioned (e.g., "I want to eat.") , in SYS \
                                 turns or if the USER requests a slot (e.g., address). The latter is tracked in the
                                 ``'state'`` dict.
                        'state': Dict
                    },
                    ...
                ],
            'speaker': 'USER' or 'SYSTEM',
            'utterance': str,

            }

        The ``'actions'`` dictionary has structure::

            {
            'act': str (name of the act, e.g., INFORM_INTENT(intent=findRestaurant), REQUEST(slot))
            'canonical_values': [str] (name of the acts). It can be the same as value for non-categorical slots. Empty
                for some acts (e.g., GOODBYE)
            'slot': str, (name of the slot that parametrizes the action, e.g., 'city'. Can be "" (e.g., GOODBYE())
            'values': [str], (value of the slot, e.g "San Jose"). Empty for some acts (e.g., GOODBYE()), or if the user
                makes a request (e.g., REQUEST('street_address'))
            }

        When the user has specified all the constraints (e.g., restaurant type and location), the next ``'SYSTEM'`` turn
        has the following _additional_ keys of the ``'actions'`` dictionary:

            {
            'service_call': {'method': str, same as the intent, 'parameters': {slot:value} specified by user}
            'service_result': [Dict[str, str], ...] where each Dict maps properties of the entity retrieved to their
                vals. Structure depends on the entity retrieved.
            }

        The dicts of the ``'slots'`` list have structure:

            {
            'exclusive_end': int (char in ``turn['utterance']`` where the slot value ends)
            'slot': str, name of the slot
            'start': int (char in ``turn['utterance']`` where the slot value starts)
            }

        The ``'state'`` dictionary has the structure::

            {
            'active_intent': str, name of the intent active at the current turn,
            'requested_slots': [str], slots the user requested in the current turn
            'slot_values': Dict['str', List[str]], mapping of slots to values specified by USER up to current turn
            }

    Returns
    -------
    Actions in the current dialogue turn.
    """

    if len(turn['frames']) > 1:
        raise IndexError("Found a more than one frame per turn!")

    actions = turn['frames'][0]['actions']
    formated_actions = []
    for d in actions:
        # acts without parameters (e.g., goodbye)
        slot = d['slot'] if d['slot'] else ''
        val = ''
        if slot:
            val = ' '.join(d['values']) if d['values'] else ''

        if slot and val:
            formated_actions.append(f"{d['act']}({slot}={val})")
        else:
            formated_actions.append(f"{d['act']}({slot})")
    return formated_actions


def print_turn_outline(outline: List[str]):
    """
    Parameters
    ----------
    outline
        Output of `get_actions`.
    """

    print(*outline, sep='\n')
    print("")


def get_dialogue_outline(dialogue: Dict) -> List[List[str]]:
    """
    Retrieves the dialogue outline, consisting of USER and SYSTEM acts, which are optionally parameterized by slots
    or slots and values.

    Parameters
    ----------
    dialogue
        Has the following structure::

            {
            'dialogue_id': str,
            'services': [str, ...], services (or APIs) that comprise the dialogue,
            'turns': [Dict[Literal['frames', 'speaker', 'utterance'], Any], ...], turns with annotations. See `get_actions`
                function for the structure of each element of the list.
            }

    Returns
    -------
    outline
        For each turn, a list comprising of the dialogue acts (e.g., INFORM, REQUEST) in that turn along with their
        parameters (e.g., 'food'='mexican', 'address').
    """
    outline = []
    for i, turn in enumerate(dialogue['turns'], start=1):
        actions = get_actions(turn)
        outline.append(actions)
    return outline


def print_dialogue_outline(dialogue: Dict, text: bool = False):
    """
    Parameters
    ----------
    dialogue
        See `get_dialogue_outline` for structure.
    text:
        If `True`, also print the utterances alongside their outlines.
    """
    outlines = get_dialogue_outline(dialogue)
    utterances = get_utterances(dialogue) if text else [''] * len(outlines)
    for i, (outline, utterance) in enumerate(zip(outlines, utterances)):
        print(f"Turn: {i}:{utterance}")
        print_turn_outline(outline)

if __name__ == '__main__':
    file = 'train/dialogues_001.json'

    with open(file, 'r') as f:
        all_dialogues = json.load(f)

    # print a random dialogue outline and its turns
    # NB: This does not work correctly for multiple frames in the same turn
    dialogue = all_dialogues[np.random.randint(0, high=len(all_dialogues))]
    print_dialogue(dialogue)
    print("")
    print_dialogue_outline(dialogue, text=True)
    print("")