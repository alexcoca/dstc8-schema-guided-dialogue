from _generate_metadata import (
    get_intents_by_type,
    get_schema_intents,
    filter_by_intent_type,
    get_entity_slots_map
)
from .test_utils import count_intents, get_random_split
from data_utils import random_sampler, dialogue_iterator

import metadata
import pytest
import random

SPLIT_NAMES = ['train', 'test', 'dev']
N_INTENTS = count_intents()


@pytest.mark.parametrize('splits', [SPLIT_NAMES, ])
def test_get_schema_intents(splits):

    for split in splits:
        intents = get_schema_intents(split)
        assert not intents['search'].intersection(intents['transactional'])


def test_get_intents_by_type():

    intents = get_intents_by_type()

    for intent_type, intent_names in intents.items():
        assert isinstance(intent_names, list)

    assert N_INTENTS == len(intents['transactional']) + len(intents['search'])


# number of dialogues to sample randomly from a randomly chosen split
n_dialogues = 100


@pytest.mark.parametrize('transactional', [True, False], ids='transactional={}'.format)
@pytest.mark.parametrize('search', [True, False], ids='search={}'.format)
@pytest.mark.parametrize('n_dialogues', [n_dialogues, ], ids='n_dialogues={}'.format)
@pytest.mark.parametrize('split', [get_random_split(), ], ids='split={}'.format)
def test_filter_by_intent_types(transactional, search, n_dialogues, split):

    # select `n_dialogues` of each dialog type at random
    transactional_dials = random.sample(
        metadata.TRANSACTIONAL_DIALOGUES[split],
        min(n_dialogues, len(metadata.TRANSACTIONAL_DIALOGUES[split])))
    search_dials = random.sample(
        metadata.SEARCH_DIALOGUES[split],
        min(n_dialogues, len(metadata.SEARCH_DIALOGUES[split])))
    mixed_intent_dials = random.sample(
        metadata.MIXED_INTENT_DIALOGUES[split],
        min(n_dialogues, len(metadata.MIXED_INTENT_DIALOGUES[split])))

    if not transactional and not search:
        with pytest.raises(ValueError) as e:
            _ = filter_by_intent_type(
                split,
                transactional=transactional,
                search=search)
            assert e.type is ValueError
    else:
        fnames_to_dials = filter_by_intent_type(
            split,
            transactional=transactional,
            search=search
        )
        all_dials = set().union(*fnames_to_dials.values())
        if search and not transactional:
            assert set(search_dials).issubset(all_dials)
        elif transactional and not search:
            assert set(transactional_dials).issubset(all_dials)
        else:
            assert set(mixed_intent_dials).issubset(all_dials)


# number of dialogues to sample randomly from a randomly chosen split
n_dialogues = 1000


@pytest.mark.parametrize('n_dialogues', [n_dialogues, ], ids='n_dialogues={}'.format)
@pytest.mark.parametrize('split', [get_random_split(), ], ids='split={}'.format)
@pytest.mark.parametrize('trials', [4, ], ids='trials={}'.format)
def test_get_entity_slots_map(n_dialogues, split, trials):

    expected_output = get_entity_slots_map()
    for _ in range(trials):
        for dial in random_sampler(split, n_dialogues):
            # output of sampler is (filename, dial)
            dial_id = dial[1]['dialogue_id']
            mixed_intent = dial_id in metadata.MIXED_INTENT_DIALOGUES
            search = dial_id in metadata.SEARCH_DIALOGUES
            if mixed_intent or search:
                for turn in dialogue_iterator(dial, user=False, system=True):
                    frame = turn['frames'][0]
                    intent = frame['service_call']['method']
                    if ('service_call' in frame) and (intent in metadata.SEARCH_INTENTS):
                        if 'service_results' in frame:
                            expected = expected_output[frame['service']][intent]
                            actual = set(slot_dict['slot'] for slot_dict in frame['slots'])
                            assert not expected - actual
