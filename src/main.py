# ruff: noqa: E402

import logging
from pathlib import Path
import argparse
import sys
import numpy as np
import json

import pandas as pd
from transformers import AutoModel, AutoTokenizer
import spacy
from spacy.tokens import Doc, Token, Span

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.utils import (
    read_input_file,
    process_text,
    produce_embedding,
    get_rotation_matrix,
    calculate_token_scores,
    get_standardization_stats,
    apply_standardization,
    align,
    add_trait,
    print_parse_tree,
    getHead,
    getDepth,
    getAdjustedDepth,
    getRight,
    firstLeftSister,
    isLeftEdge,
    hasLeftChildren,
    isRoot,
    takesBareInfinitive,
    tensed_clause,
    getTensedVerbHead,
    pastThreshold,
    meanAbove,
    dimHighlight,
    negHighlight,
    apply_highlights,
    get_html_string)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

trait_vector = {'Formality': (10, 132, 1, 'nospread', {'DEFAULT': 1.7}, None, 'PaleGreen', 'PaleGreen', 'White', 'White', False),
                'Narrativity': (6, 210, 1, 'spread',
                                {'ADP': 2, 'DET': 2, 'AUX': 0.4, 'ADV': 0.5, 'PRON': 0.8, 'PROPN': 0.5, 'VERB': 0.8, 'DEFAULT': 1},
                                {'QUOTE': -.5, 'PRON': -1, 'DEFAULT': -2},'MediumSeaGreen', 'PaleGreen', 'White', 'White', False),
                'SentenceComplexity': (0, 190, 1, 'multilevel', {'DEFAULT': 0.7, 'SUBRULE': {'DEFAULT': 2}}, {'DEFAULT': -.7, 'SUBRULE': {'DEFAULT': -1}}, 'MediumSeaGreen', 'PaleGreen', 'DarkRed', 'LightCoral', False),
                'Cohesion': (10, 210, 1, 'nospread', {'PUNCT': 5, 'DEFAULT': 2}, None, 'PaleGreen', 'PaleGreen', 'White', 'White', False),
                'Dialog': (8, 347, -1, 'fullspread', {'QUOTE': 0.5, 'PRON': 1.2, 'ADP': 1.2, 'DEFAULT': 1}, None, 'MediumSeaGreen', 'PaleGreen', 'White', 'White', False),
                'Organization': (11, 348, 1, 'multilevel', {'PUNCT': 2, 'DET': 2, 'DEFAULT': 0, 'SUBRULE': {'DEFAULT': 1}}, {'DEFAULT': 0, 'SUBRULE': {'DEFAULT': -1}}, 'MediumSeaGreen', 'PaleGreen', 'DarkRed', 'LightCoral', False),
                'GrammarUsage': (11, 164, -1, 'spread', None, {'PUNCT': -2, 'DET': -2, 'ADP': -1, 'NOUN': -0.1, 'PROPN': -0.1, 'DEFAULT': -.5}, 'White', 'White', 'LightCoral', 'LightCoral', False),
                'Concreteness': (4, 315, -1, 'nospread', {'PUNCT': 2, 'ADJ': 1.5, 'ADP': 0.8, 'DEFAULT': 1.3}, None, 'PaleGreen', 'PaleGreen', 'White', 'White', False),
                'Conventionality': (10, 285, 1, 'nospread', None, {'ADP': -2.5, 'SCONJ': -1, 'PRON': -.5, 'CCONJ': -1, 'X': -1, 'INTJ': -1.5, 'PUNCT': -2, 'DEFAULT': -3.3}, 'White', 'White', 'IndianRed', 'LightCoral', False),
                'Interactivity': (10, 362, -1, 'spread', {'PUNCT': 2, 'NOUN':1, 'PROPN': 1.5, 'ADJ': 1, 'NUM': 1, 'ADV': 0.5, 'VERB': 0.8, 'SCONJ': 1, 'CCONJ': 1, 'AUX': 1.3, 'DEFAULT': 0.3}, {'DET': -2, 'PRON': -2, 'PUNCT': -2, 'AUX': -2, 'DEFAULT': -1}, 'PaleGreen', 'PaleGreen', 'White', 'White', False),
                'Contextualization': (10, 107, 1, 'nospread', {'PRON': 0.3, 'PROPN': 0.5, 'PUNCT': 1.5, 'DEFAULT': 0.95}, None, 'MediumSeaGreen', 'PaleGreen', 'White', 'White', True),
                'StanceTaking': (0, 373, -1, 'spread', {'AUX': 2.5, 'PRON': 2, 'DET': 2, 'CCONJ': 1.5, 'SCONJ': 1.5, 'DEFAULT': 1.3}, None, 'MediumSeaGreen', 'PaleGreen', 'White', 'White', False),
                'VocabularyDifficulty': (10, 132, 1, 'nospread', {'DEFAULT': 2}, None, 'PaleGreen', 'PaleGreen', 'White', 'White', False),
                'hscore': (7, 358, -1, 'spread', {'PUNCT': 2, 'DEFAULT': 1}, None, 'PaleGreen', 'PaleGreen', 'White', 'White', False),
                'VocabularyFrequency': (5, 187, 1, 'nospread', {'PUNCT': 2, 'DEFAULT': 0}, None, 'PaleGreen', 'PaleGreen', 'White', 'White', False),
                'LexicalTightness': (10, 35, -1, 'nospread', {'PRON': 2, 'ADJ': 1.2, 'PUNCT': 2, 'CCONJ': 2, 'ADV': 1.5, 'AUX': 2, 'DEFAULT': 0.9}, None, 'PaleGreen', 'PaleGreen', 'White', 'White', False)}

excludes = [3,75,76,81,132,178,206,244,245,304,367,373]

def main(
    input_text_path: str,
    rotation_path: str,
    output_path: str,
    text_id: str,
    text: str,
    passthrough: str,
    csv_input: bool=False,
    csv_path: bool=False,
    html_path: bool=False
):
    rotation_matrix = get_rotation_matrix(rotation_path)

    if csv_input:
        csvFile = pd.read_csv(input_text_path)
    else:
        essay = read_input_file(input_text_path)
        text_id = 'ID'
        text = 'full_essay'
        csvFile = pd.DataFrame([[input_text_path, essay]])
        csvFile.columns = [text_id, text]

    rows = []
    for index, row in csvFile.iterrows():
        tID = row[text_id]
        print('processing', tID)

        if passthrough is not None:
            passthrough_vars = passthrough.split(',')

        essay = row[text]
        if essay is None or type(essay) is float:
            continue
        essay = essay.replace('."', '".')
        
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(essay)
        Token.set_extension('dim_highlight', default=False, force=True)
        Token.set_extension('bright_highlight', default=False, force=True)
        Token.set_extension('neg_highlight', default=False, force=True)
        Token.set_extension('bright_neg_highlight', default=False, force=True)

        tokens = tokenizer.tokenize(essay)
        tokens = [token.replace("▁", "") for token in tokens]
        
        per_layer_hidden_states = process_text(essay, model, tokenizer)
        embeddings = produce_embedding(essay, model, tokenizer)           
        output = {}

        for trait in trait_vector:
            if csv_input:
                output['ID'] = tID

            (layer, dimension, sign, method, rule, negRule, color1, color2, color3, color4, include_stops) = trait_vector[trait]
            trait_score = np.matmul(embeddings[layer][0], rotation_matrix[layer])[dimension]*sign
 
            token_scores = calculate_token_scores(
                per_layer_hidden_states, rotation_matrix, layer, dimension
            )
            token_scores = token_scores*sign

            if html_path and not csv_input:
                html_string = get_html_string(doc, tokens, token_scores, trait, method, rule, color1, color2, negRule, color3, color4, include_stops)
                outfile = output_path + '_' + trait + '.html'
                with open(outfile, "w") as text_file:
                    text_file.write(html_string)

            token_scores = list(zip(tokens, token_scores))
            if csv_input:
                output[trait] = trait_score
                for var in passthrough_vars:
                    if var in row:
                        output[var] = row[var]
            else:
                output[trait] = (trait_score, token_scores)

            if csv_path and not csv_input:
                outname = output_path.replace('.csv','').replace('.json','') + '_' + trait + '.csv'
                ts = pd.DataFrame(token_scores, columns=['Token', 'Score'])
                ts.to_csv(outname)

        final_layer = np.matmul(embeddings[12][0], rotation_matrix[12])
        for i in range(0,len(final_layer)):
            if i not in excludes:
                output['V' + str(i)] = final_layer[i]

        if csv_input:
            rows.append(output)
        else:
            with open(output_path + '.json', "w") as data_file:
                json.dump(output, data_file, indent=4, sort_keys=False)

    if csv_input:
        df = pd.DataFrame(rows)
        df.to_csv(output_path + '.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file")
    parser.add_argument("-r", "--rotation_path")
    parser.add_argument("-o", "--output_path")
    parser.add_argument("-d", "--text_id", default='ID')
    parser.add_argument("-t", "--text", default='full_text')
    parser.add_argument("-s", "--passthrough", default='holistic_essay_score')
    parser.add_argument("-v", "--csv_input", action="store_true")
    parser.add_argument("-c", "--csv_path", action="store_true")
    parser.add_argument("-p", "--html_path", action="store_true")
    args = parser.parse_args()

    model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall")

    main(
        args.input_file,
        args.rotation_path,
        args.output_path,
        args.text_id,
        args.text,
        args.passthrough,
        args.csv_input,
        args.csv_path,
        args.html_path,
    )
