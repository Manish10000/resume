import re
import json
import logging
import spacy
import random
from pathlib import Path
from spacy.training import Example
from spacy.util import minibatch,compounding

def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """
    if data is None:
        return []
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])

    return cleaned_data
def convert_dataturks_to_spacy(dataturks_JSON_FilePath):
    try:
        training_data = []
        lines = []
        with open(dataturks_JSON_FilePath, 'r', encoding="utf8") as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            if data['annotation'] is not None:
                for annotation in data['annotation']:
                    # only a single point in text annotation.
                    point = annotation['points'][0]
                    labels = annotation['label']
                    # handle both list of labels or a single label.
                    if not isinstance(labels, list):
                        labels = [labels]

                    for label in labels:
                        # dataturks indices are both inclusive [start, end]
                        # but spacy is not [start, end)
                        entities.append((
                            point['start'],
                            point['end'] + 1,
                            label
                        ))
            entities=merge_overlapping_entities(entities)
            training_data.append((text, {"entities": entities}))
        return training_data
    except Exception:
        logging.exception("Unable to process " + dataturks_JSON_FilePath)
        return None
def merge_overlapping_entities(entities):
    entities.sort(key=lambda x: x[0])  # Sort entities by start offset
    merged_entities = []
    current_entity = None

    for start, end, label in entities:
        if current_entity is None or start > current_entity[1]:
            # No overlap with the current entity, add a new one
            current_entity = (start, end, label)
            merged_entities.append(current_entity)
        else:
            # There is an overlap, merge entities
            current_entity = (current_entity[0], max(end, current_entity[1]), label)
            merged_entities[-1] = current_entity

    return merged_entities
TRAIN_DATA = trim_entity_spans(convert_dataturks_to_spacy("traindata.json"))

def main(model=None, new_model_name="training", output_dir='/Users/mrmjpatra/Documents/SmalldayTech/OnBoardingResume/parser1', n_iter=30):
    if model is not None:
        nlp = spacy.load(model)  # load existing spacy model
        print(f"Loaded model '{model}'")
    else:
        nlp = spacy.blank("en")  # create blank language class
        print("Created blank 'en' model")

    # add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spacy
    if "ner" not in nlp.pipe_names:
        print("Creating new pipe")
        ner = nlp.add_pipe("ner")
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # if model is None or reset_weights:
    #     optimizer = nlp.begin_training()
    # else:
    #     optimizer = nlp.resume_training()

    optimizer = nlp.initialize()

    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    
    with nlp.select_pipes(enable=["ner"]):
        # batch up the examples using spacy's minibatch
        for itn in range(n_iter):
            print(f"Starting iteration {itn}")
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                example = [Example.from_dict(nlp.make_doc(text), annotation) for text, annotation in zip(texts, annotations)]
                nlp.update(example, drop=0.2, sgd=optimizer, losses=losses)
            print("Losses", losses)

    test_text = "marathwada mitra mandals college of engineering"
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)

main()