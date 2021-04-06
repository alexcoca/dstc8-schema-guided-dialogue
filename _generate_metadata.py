import subprocess

from analysis import \
    get_intents_by_type, \
    get_requestable_slots, \
    get_binary_slots, \
    get_entity_slots_map,\
    _map_intents_to_services

import json

def get_commit_hash():
    """Returns the commit hash for the current HEAD."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()




if __name__ == '__main__':

    # find transactional and search/query intents in the corpus
    _intents_by_type = get_intents_by_type()
    # find binary slots and their breakdown by service
    binary_slots = get_binary_slots()

    metadata = {
        'VERSION': get_commit_hash(),
        'ALL_INTENTS': _intents_by_type['search'].union(_intents_by_type['transactional']),
        'SEARCH_INTENTS': _intents_by_type['search'],
        'TRANSACTIONAL_INTENTS': _intents_by_type['transactional'],
        'INTENTS_TO_SERVICES': _map_intents_to_services(),
        'REQUESTABLE_SLOTS': get_requestable_slots(),
        'BINARY_SLOTS': binary_slots[0],
        'BINARY_SLOTS_BY_SERVICE': binary_slots[1],
        'ENTITY_SLOTS': get_entity_slots_map(),
    }

    with open('metadata.json', 'w') as f:
        json.dumps(metadata)