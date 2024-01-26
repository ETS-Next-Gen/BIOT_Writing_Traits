# ruff: noqa: E402

import logging
from pathlib import Path
import argparse
import sys
import numpy as np
import json
import string

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
    get_html_string)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main(
    input_text_path: str,
    rotation_path: str,
    output_path: str,
    text_id: str,
    text: str,
    passthrough: str,
    highlight_path: str,
    csv_input: bool=False,
    csv_path: bool=False,
    html_path: bool=False,
    embeddings: bool=False,
    no_rotation: bool=False
):
    """
        This is the main program for the BIOT trait model
        which supports several functionalities:

        1. Outputting raw embeddings and associated
        predictors for the purpose of running BIOT
        rotations on them.

        2. Calculating and outputting trait scores at the
        essay summary level, using rotation mnatrices
        calculated using BIOT.

        3. Calculating trait scores by token and highlighting
        tokens to show high or low scores on a trait, using a
        rule base that specifies highligting methods, cutoffs
        for highlighting and colors to be highlighted.
    """
    
    # load our rotation matrices we'll use to calculate
    # trait scores
    rotation_matrix = get_rotation_matrix(rotation_path)
    
    # load the highlighting rules to token trait score info
    with open(highlight_path + 'trait_vector.json',
              "r") as tvec:
        trait_vector = json.load(tvec)
    with open(highlight_path + 'excludes.json',
              "r") as excl:
        excludes = json.load(excl)

    # consistency check
    if embeddings and not csv_input:
        print('Supply CSV file containing input texts')
 
    # set up input data appropriately depending
    # on whether it's one file or a list of essays
    # in a csv file.
    if csv_input:
        csvFile = pd.read_csv(input_text_path)
    else:
        essay = read_input_file(input_text_path)
        text_id = 'ID'
        text = 'full_essay'
        csvFile = pd.DataFrame([[input_text_path,
                                 essay]])
        csvFile.columns = [text_id, text]

    # set up output variables
    rows = []
    embedding_output = {}
    ids = []
    predictor_output = []
    
    for index, row in csvFile.iterrows():
        
        # record the ID for the specific text being processed
        tID = row[text_id]
        print('processing',
              tID,
              ' ',
              index,
              'of',
              len(csvFile))

        # We use passthrough_vars to indicate fields in the input csv
        # file that are to be output unchanged in the output file.
        if passthrough is not None:
            passthrough_vars = passthrough.split(',')

        # record the input essay text using the field we've assigned
        # as holding that string.
        essay = row[text]

        if isinstance(essay, float):
            continue

        # small adjustment to make sure quotation marks
        # are correctly associated with the sentence they
        # belong with
        essay = essay.replace('."', '".')

        if len(essay.translate({ord(c): None for c in string.whitespace}))==0:
            continue
        # If we're going to output highlighted text, we'll be using
        # a spacy parse to manage the process, so set up Spacy properly
        nlp = None
        doc = None
        if output_path and not no_rotation and html_path:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(essay)
            Token.set_extension('dim_highlight',
                                default=False,
                                force=True)
            Token.set_extension('bright_highlight',
                                default=False,
                                force=True)
            Token.set_extension('neg_highlight',
                                default=False,
                                force=True)
            Token.set_extension('bright_neg_highlight',
                                default=False,
                                force=True)

        # get the list of tokens needed to match the data
        # produced by the LLM
        tokens = tokenizer.tokenize(essay)
        tokens = [token.replace("▁",
                                "") for token in tokens]
        
        # get the hidden states out of the LLM
        per_layer_hidden_states = process_text(essay,
                                               model,
                                               tokenizer)
        embedding_matrix = produce_embedding(essay,
                                             model,
                                             tokenizer)

        # if we need to produce the unmodified embeddings (for instance,
        # to run BIOT to calculate rotation matrices), gather the
        # necessary information
        if embeddings and csv_input:
            ids.append(tID)
            for index, layer in enumerate(embedding_matrix):
                output = {}
                for i in range(0,len(layer[0])):
                    output['V' + str(i)] = layer[0][i]
                if index not in embedding_output:
                    embedding_output[index] = []
                embedding_output[index].append(output)

            predictors = {}
            for varName in passthrough_vars:
                predictors[varName] = row[varName]
            predictor_output.append(predictors)

        # if we're going to apply rotations and output something, do
        # this main code segment
        if output_path and not no_rotation:
            output = {}

            # apply the same logic to every trait
            for trait in trait_vector:

                # make sure the ID field is present in the  output
                if csv_input:
                    output['ID'] = tID                

                # retrieve the rule relevant to this trait from trait_vector
                (layer, dimension, sign, method, rule, negRule, color1,
                 color2, color3, color4, include_stops) = trait_vector[trait]

                # calculate the essay's score for this particular trait 
                # Note we reverse sign of the trait score if the rule says
                # so (sign should always be 1 or -1)
                trait_score = \
                    np.matmul(embedding_matrix[layer][0],
                              rotation_matrix[layer])[dimension]*sign
 
                # calculate the individual trait scores for each token            
                token_scores = calculate_token_scores(
                    per_layer_hidden_states,
                    rotation_matrix,
                    layer,
                    dimension
                )
                # reverse the sign of the token score if the rule says so
                token_scores = token_scores*sign

                # if we want to produce highlighted outputs to
                # visualize the token scores, pass in the rule
                # information for the trait and get back the
                # highlighting
                if html_path and not csv_input:
                    html_string = get_html_string(doc,
                                                  tokens,
                                                  token_scores,
                                                  trait,
                                                  method,
                                                  rule,
                                                  color1,
                                                  color2,
                                                  negRule,
                                                  color3,
                                                  color4,
                                                  include_stops)
                    # then save the highlighted file
                    outfile = output_path + '_' + trait + '.html'
                    with open(outfile, "w") as text_file:
                        text_file.write(html_string)

                # reformat the token scores so they're
                # explicitly associated with tokens
                if not isinstance(token_scores, list):
                    token_scores = [token_scores] 
                token_scores = list(zip(tokens,
                                        token_scores))
                # add the traits to the output matrix
                if csv_input:
                    output['biot_' + trait] = trait_score
                    for var in passthrough_vars:
                        if var in row:
                            output[var] = row[var]
                else:
                    output[trait] = (trait_score,
                                     token_scores)

                # save the individual trait datafile if we're
                # only processing one file
                if csv_path and not csv_input:
                    outname = \
                        output_path.replace('.csv',
                                            '').replace('.json',
                                                        '') \
                                                        + '_' \
                                                        + trait \
                                                        + '.csv'
                    ts = pd.DataFrame(token_scores,
                                      columns=['Token',
                                               'Score'])
                    ts.to_csv(outname, index=False)

            # get the final layer's  hidden state, since we'll
            # want to keep that information in our output.
            final_layer = np.matmul(embedding_matrix[12][0],
                                    rotation_matrix[12])

            # exclude dimensions that correspond to the
            # traits we've calculated already
            for i in range(0,len(final_layer)):
                if i not in excludes:
                    output['V' + str(i)] = final_layer[i]

            # output the token score data for each trait
            if csv_input:
                rows.append(output)
            else:
                with open(output_path + '.json',
                          "w") as data_file:
                    json.dump(output,
                              data_file,
                              indent=4,
                              sort_keys=False)

    # output the overall trait score output if required
    if csv_input and output_path and not no_rotation:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    # output the original embeddings if required
    if embeddings and csv_input and output_path:

        df0 = pd.DataFrame(ids)
        df0.to_csv(output_path + '_IDs.csv',
                   index=False,
                   header=False)
        
        for layerID in embedding_output:
            df = pd.DataFrame(embedding_output[layerID])
            df.to_csv(output_path.replace('.csv', '') \
                      + '_embedding_layer_' + \
                      str(layerID) + '.csv',
                      index=False,
                      header=False)
            
        df2 = pd.DataFrame(predictor_output)
        df2.to_csv(output_path.replace('.csv',
                                       '') \
                   + '_predictors.csv',
                   index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file")
    parser.add_argument("-r", "--rotation_path")
    parser.add_argument("-o", "--output_path")
    parser.add_argument("-d", "--text_id",
                        default='ID')
    parser.add_argument("-t", "--text",
                        default='full_text')
    parser.add_argument("-s", "--passthrough",
                        default='holistic_essay_score')
    parser.add_argument("-v", "--csv_input",
                        action="store_true")
    parser.add_argument("-c", "--csv_path",
                        action="store_true")
    parser.add_argument("-p", "--html_path",
                        action="store_true")
    parser.add_argument("-x", "--highlight_path",
                        default="data/rotation/highlighting/")
    parser.add_argument("-e", "--embeddings",
                        action="store_true")
    parser.add_argument("-n", "--no_rotation",
                        action="store_true")
    args = parser.parse_args()

    # load our LLM and tokenizer
    model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall")

    # then run the main program
    main(
        args.input_file,
        args.rotation_path,
        args.output_path,
        args.text_id,
        args.text,
        args.passthrough,
        args.highlight_path,
        args.csv_input,
        args.csv_path,
        args.html_path,
        args.embeddings,
        args.no_rotation
    )
