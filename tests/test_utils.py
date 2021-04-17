from data_utils import schema_iterator

import random

SPLIT_NAMES = ['train', 'dev', 'test']


def count_intents():

    intents = set()
    for split in SPLIT_NAMES:
        for service in schema_iterator(split):
            for intent_dict in service['intents']:
                intents.add(intent_dict['name'])

    return len(intents)


def get_random_split():
    return random.choice(SPLIT_NAMES)
