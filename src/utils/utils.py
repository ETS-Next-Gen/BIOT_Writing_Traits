import typing as t
from pathlib import Path
import glob
import re
import json

import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import spacy
from spacy.tokens import Doc, Token, Span

import nltk
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))

def read_input_file(input_path: str) -> str:
    assert input_path.endswith(".txt")
    with open(Path(input_path), "r") as f:
        text = f.read()

    return text


def process_text(
    input_str: str, model: AutoModel, tokenizer: AutoTokenizer
) -> t.List[np.ndarray]:
    """
    Process input text using the provided model and tokenizer.
    """
    inputs = tokenizer(input_str, add_special_tokens=False, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)
    per_layer_hidden_states = [
        hidden_states.detach().numpy() for hidden_states in outputs.hidden_states
    ]

    return per_layer_hidden_states

def produce_embedding(
    input_str: str, model: AutoModel, tokenizer: AutoTokenizer
) -> t.List[np.ndarray]:
    """
    Process input text using the provided model and tokenizer.
    """
    inputs = tokenizer(input_str, add_special_tokens=False, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)
    per_layer_hidden_states = [
        torch.mean(hidden_states.detach().cpu(), dim=1).numpy() for hidden_states in outputs.hidden_states
    ]

    return per_layer_hidden_states


def get_rotation_matrix(path: str = None) -> t.Dict:
    """ """
    out = {}
    matrix_paths = glob.glob(str(Path(path) / "*.csv"))
    for matrix_path in matrix_paths:

        rotation_matrix = pd.read_csv(matrix_path, header=None).to_numpy()
        i_layer = re.search(r"layer_([\d]+)_", str(matrix_path)).group(1)
        out[int(i_layer)] = rotation_matrix

    return out


def calculate_token_scores(
    hidden_state: np.ndarray,
    rotation_matrix_dict: t.Dict,
    i_layer: int = -1,
    i_dim: int = 0,
) -> np.ndarray:
    """
    Calculate scores for tokens based on hidden states and rotation matrix.
    """
    layer_state = hidden_state[int(i_layer)]
    rotation_matrix = rotation_matrix_dict[int(i_layer)]

    assert layer_state.shape[-1] == rotation_matrix.shape[0]

    scores = np.matmul(layer_state, rotation_matrix)[:, :, int(i_dim)]
    scores = scores.squeeze()

    return scores


def get_standardization_stats(path: str, i_layer: int) -> t.Dict:
    with open(Path(path), "r") as f:
        per_layer_stats = json.load(f)

    return per_layer_stats[str(i_layer)]


def apply_standardization(
    score_matrix: np.ndarray, mean: float, std: float
) -> np.ndarray:
    return (score_matrix - mean) / std


def align(doc, tokens, scores, aligned_tokens, aligned_scores):
    if len(doc) == 0 or len(tokens)==0:
        return aligned_tokens, aligned_scores
    if doc[0].text.lower() == tokens[0].lower():
        if len(tokens)>0 and len(scores)>0:
            aligned_tokens.append(tokens[0])
            aligned_scores.append(scores[0])
        return align(doc[1:], tokens[1:], scores[1:], aligned_tokens, aligned_scores)
    else:
        loc = 1
        combotok = tokens[0].lower() + tokens[loc].lower()
        if tokens[0].lower().startswith(doc[0].text.lower()):
            aligned_scores.append(scores[0])
            aligned_tokens.append(doc[0].text.lower())
            leftover = tokens[0][len(doc[0].text):]
            tokens[1] = leftover + tokens[1]
            return align(doc[1:], tokens[1:], scores[1:], aligned_tokens, aligned_scores)
        
        while loc<len(tokens) and combotok != doc[0].text.lower() and len(combotok) <= len(doc[0].text):
            loc+=1
            combotok += tokens[loc].lower()
        if len(combotok) > len(doc[0].text):
            aligned_tokens.append(doc[0].text)
            aligned_scores.append(0)
            return align(doc[1:], tokens, scores, aligned_tokens, aligned_scores)
        elif combotok == doc[0].text.lower():
            aligned_tokens.append(combotok)
            total = 0
            for i in range(0,loc+1):
                total += scores[i]
            aligned_scores.append(total/(loc+1))
            return align(doc[1:], tokens[loc+1:], scores[loc+1:], aligned_tokens, aligned_scores)
        else:
            print('mismatch')
            return align(doc[1:], tokens[1:], scores[1:], aligned_tokens, aligned_scores)

def add_trait(doc, trait, aligned_scores):
    Token.set_extension(trait, default=0)
    for i, token in enumerate(aligned_scores):
        setattr(doc[i]._, trait, aligned_scores[i])

def print_parse_tree(sent, trait):
    """
        Print pretty formatted version of parse tree
    """

    lastToken = None
    headLoc = 0
    lastHeadLoc = 0

    ########################################################################
    # print each token in the sentence in turn with appropriate annotation #
    ########################################################################
    for token in sent:

        # set up some useful variables
        head = getHead(token)
        depth = getDepth(token)
        rightNeighbor = getRight(sent, token.i)

        # the actual depth we want to indent, as opposed to depth
        # in the parse tree
        usedDepth = getAdjustedDepth(token)

        ##############################################################
        # output the basic category status of the phrasal elements   #
        ##############################################################
        cat = ''

        # special case -- punctuation
        if token.dep_ == 'advmod' \
           and rightNeighbor is not None \
           and rightNeighbor.dep_ == 'punct':
            if token.tag_.lower().startswith('R'):
                cat = 'RB'
            else:
                cat = 'AP'

        # special case -- gerunds
        elif (token.dep_ == 'xcomp'
              and token.tag_ == 'vbg'):
            cat = 'SG'

        # special case -- auxiliaries at depth zero in the parse tree
        elif (depth == 0
              and (token.tag_ == 'BEZ'
                   or token.tag_ == 'BEM'
                   or token.tag_ == 'BER'
                   or token.tag_.startswith('HV')
                   or token.tag_.startswith('DO'))):
            cat = 'VP'

        # main branch of logic. The firstLeftsister function
        # helps us find the leftmost member of a np, vp, or
        # pp etc. span
        elif firstLeftSister(token, lastToken):
            if token.tag_ == 'VBG' and \
               token.dep_ == 'csubj':
                cat = 'SG'  # gerund
            elif token.tag_ == 'WRB':  # wh adverbs
                if head is not None:
                    for child in head.children:
                        if child.tag_ == 'TO':
                            cat = 'SI RB'
                            # infinitival clause with wh adverb
                        else:
                            cat = 'SB'
                            # subordinate clause with wh adverb
                else:
                    cat = 'SB'
                    # subordinate clause with wh adverb
            elif (token.tag_ == 'TO'
                  and head is not None
                  and head.dep_ == 'xcomp'):
                cat = 'SI'  # infinitive clause
            elif (token.dep_ == 'mark'
                  and head is not None
                  and (head.dep_ == 'advcl')):
                cat = 'SB\tCOMP'  # adverbial subordinate clause
            elif (token.dep_ == 'mark'
                  and head is not None
                  and (head.dep_ == 'ccomp'
                       or head.dep_ == 'acl'
                       or head.dep_ == 'csubj')):
                cat = 'SC\tCOMP'   # complement clause
            elif (token.dep_ == 'mark'
                  and head is not None
                  and head.dep_ == 'relcl'):
                cat = 'SR\tCOMP'   # relative clause with that
            elif token.tag_ == 'WDT':
                cat = 'SR\tNP'  # relative clause with wh determiner
            elif token.tag_ == 'WPS':
                cat = 'SR\tNP'  # relative clause with wh pronoun
            elif (token.tag_.startswith('V')
                  and token.dep_ == 'conj'
                  and head is not None
                  and head.dep_ == 'advcl'):
                cat = 'SB'
                # adverbial subordinate clause
                # in compound structure
            elif (token.tag_.startswith(' ')
                  and token.dep_ == 'conj'
                  and head is not None
                  and head.dep_ == 'ccomp'):
                cat = 'SC'   # complement clause in compound structure
            elif (token.tag_.startswith('V')
                  and token.dep_ == 'conj'
                  and head is not None
                  and head.dep_ == 'acl'):
                cat = 'SC'  # compound clause in compound structure
            elif (token.tag_.startswith('V')
                  and token.dep_ == 'conj'
                  and head is not None
                  and head.dep_ == 'relcl'):
                cat = 'SR'  # relative clause
            elif (token.tag_.startswith('V')
                  and token.dep_ == 'conj'
                  and head is not None
                  and head.dep_ == 'xcomp'):
                cat = 'SJ'  # conjoined main clause or VP
            elif (token.tag_ == 'CC'
                  and head is not None
                  and isRoot(head)):
                cat = 'SJ'  # conjoined main clause or VP
            elif token.tag_ == 'CC':
                cat = 'CC'  # coordinating conjunction
            elif (token.dep_ == 'prep'
                  or token.dep_ == 'agent'):
                cat = 'PP'  # prepositional phrase
            elif (token.dep_ == 'acomp'
                  or (token.dep_ == 'neg'
                      and head is not None
                      and (head.tag_.startswith('J')
                           or head.tag_.startswith('R')))
                  or (token.dep_ == 'advmod'
                      and (head is not None
                           and head.dep_ != 'amod'
                           and (head.i < token.i
                                or head.tag_.startswith('R')
                                or head.tag_.startswith('J'))))):
                if (token.tag_.lower().startswith('R')):
                    cat = 'RB'  # adverb or adverb phrase
                else:
                    cat = 'AP'  # adjective phrase
            elif (token.dep_ == 'det'
                  or (token.dep_ == 'neg'
                      and head is not None
                      and head.dep_ == 'det')
                  or token.dep_ == 'poss'
                  or token.dep_ == 'amod'
                  or token.dep_ == 'nummod'
                  or token.dep_ == 'compound'
                  or token.dep_ == 'nsubj'
                  or token.dep_ == 'nsubjpass'
                  or token.dep_ == 'dobj'
                  or token.dep_ == 'pobj'
                  or token.dep_ == 'appos'
                  or token.dep_ == 'attr'
                  or token.tag_.startswith('N')
                  or token.tag_.startswith('TUNIT')):
                cat = 'NP'  # noun phrase
            elif ((depth == 0
                   and not hasLeftChildren(token))
                  or token.dep_ == 'aux'
                  or token.dep_ == 'auxpass'
                  or token.dep_ == 'neg'
                  or (token.dep_ == 'advmod'
                      and token.i < head.i)
                  or token.dep == 'acl'
                  or token.dep_ == 'relcl'
                  or token.dep_ == 'advcl'
                  or token.dep_ == 'ccomp'
                  or token.tag_.startswith('V')
                  or token.tag_.startswith('BE')
                  or token.tag_.startswith('DO')
                  or token.tag_.startswith('HV')):
                cat = 'VP'  # verb phrase

        headLoc -= 1
        header = '\t'

        ################################################################
        # Set up the header element that captures category information #
        # and depth                                                    #
        ################################################################

        # mark the start of the sentence
        if isLeftEdge(token, sent):
            header += 'S'

        # add tabs to capture the degree of indent we are setting for
        # this word
        while usedDepth >= 0:
            header += '\t'
            usedDepth -= 1
        # put the category of the word as the first item in the indent
        header += cat

        headLoc = -1
        if head is not None:
            headLoc = head.i

        ##################################################################
        # format the whole line and print it. Index of word plus header  #
        # information including word category, followed by the token's   #
        # tag and text, its lemma in parentheses, followed by the de-    #
        # pendency label and the index of the word the dependency points #
        # to                                                             #
        ##################################################################
        line = str(token.i) \
            + header \
            + "\t|" \
            + token.tag_ \
            + " " \
            + token.text + \
            " (" \
            + token.lemma_.replace('\n', 'para') \
            + ")" \
            + " " \
            + token.dep_ + \
            ":" \
            + str(headLoc) + \
            " " + \
            trait + \
            ":" + \
            str(round(getattr(token._,trait),2))

        line = line.expandtabs(6)
        print(line)

        lastToken = token
        if head is not None:
            lastHeadLoc = head.i

def getHead(tok: Token):
    if tok is not None and tok is not bool:
        for anc in tok.ancestors:
            return anc
    return None


def getDepth(tok: Token):
    """
     This function calculates the depth of the current word
     in the spaCY dependency tree
    """
    depth = 0
    if tok is not None:
        for anc in tok.ancestors:
            depth += 1
    return depth

def getAdjustedDepth(tok: Token):
    """
     This function adjusts the depth of the word node to the
     depth we want to display in the output
    """
    depth = getDepth(tok)
    adjustment = 0
    if tok is not None:
        for anc in tok.ancestors:
            # clausal subjects need to be embedded one deeper
            # than other elements left of the head, but
            # otherwise we decrease indent of elements left of
            # the head to the indent of the head, to display
            # them as a single span
            if tok.i < anc.i \
               and anc.dep_ != 'csubj' \
               and tok.dep_ != 'csubj':
                adjustment += 1
            # clauses should be indented one level deeper
            # than the dependency tree suggests
            if tok.dep_ == 'advcl' \
               or tok.dep_ == 'ccomp' \
               or tok.dep_ == 'acl' \
               or tok.dep_ == 'relcl':
                adjustment -= 1
            if anc.dep_ == 'advcl' \
               or anc.dep_ == 'ccomp'\
               or anc.dep_ == 'acl' \
               or anc.dep_ == 'relcl':
                adjustment -= 1
    head = getHead(tok)
    if tok.dep_ == 'mark' \
       and head is not None \
       and head.dep_ == 'csubj':
        adjustment += 1
    return depth-adjustment

def getRight(sentence, loc):
    """
     This function returns the word immediately to the
     right of the input token
    """
    if loc + 1 < sentence.__len__():
        return sentence[loc + 1]
    return None


def firstLeftSister(tokenA: Token, tokenB: Token):
    """
     This function indicates that a word is the leftmost
     dependent of a head word. It requires a speries of
     special checks based upon knowledge that for instance
     'case' (possessive) and punctuations don't interrupt
     a phrase, and that each type of phrase uses a limited
     number of dependencies for left sisters
    """
    depth = getDepth(tokenA)
    depthB = getDepth(tokenB)
    head = getHead(tokenA)
    if abs(depth - depthB) > 1 \
       and (tokenB is None
            or tokenB.dep_ != 'case'
            and tokenB.dep_ != 'punct'):
        return True
    if tokenA is not None \
       and tokenB is None:
        return True
    elif (tokenA is not None
          and tokenB is not None):
        if tokenA.dep_ == 'prep' \
           and tokenB.tag_.startswith('R') \
           and tokenB.lower_.endswith('ly'):
            return True
        if tokenA.dep_ == 'advmod' \
           and tokenA.tag_.startswith('R') \
           and head.tag_.startswith('V') \
           and head.i == tokenA.i - 1:
            return True
        if (tokenA.dep_ == 'aux'
            or tokenA.dep_ == 'auxpass'
            or tokenA.dep_ == 'neg'
            or tokenA.dep_ == 'advmod'
            or tokenA.dep_ == 'advcl'
            or tokenA.dep_ == 'relcl'
            or tokenA.dep_ == 'conj'
            or tokenA.tag_.startswith('V')
            or tokenA.tag_.startswith('BE')
            or tokenA.tag_.startswith('DO')
            or tokenA.tag_.startswith('HV')) \
           and (tokenB.dep_ == 'aux'
                or tokenB.dep_ == 'auxpass'
                or tokenB.dep_ == 'neg'
                or tokenB.dep_ == 'advmod'
                or (tokenB.dep_ == 'punct'
                    and tokenA in tokenB.ancestors)
                or tokenB.tag_.startswith('BE')
                or tokenB.tag_.startswith('DO')
                or tokenB.tag_.startswith('HV')
                or tokenB.tag_.startswith('V')):
            return False
        if (tokenA.dep_ == 'det'
            or tokenA.dep_ == 'poss'
            or tokenA.dep_ == 'amod'
            or tokenA.dep_ == 'nummod'
            or tokenA.dep_ == 'compound'
            or tokenA.dep_ == 'nsubj'
            or tokenA.dep_ == 'nsubjpass'
            or tokenA.dep_ == 'csubj'
            or tokenA.dep_ == 'csubjpass'
            or tokenA.dep_ == 'dobj'
            or tokenA.dep_ == 'pobj'
            or tokenA.dep_ == 'attr'
            or tokenA.dep_ == 'appos'
            or (tokenA.dep_ == 'neg'
                and getHead(tokenA) is not None
                and getHead(tokenA).dep_ == 'det')) \
           and (tokenB.dep_ == 'det'
                or tokenB.dep_ == 'poss'
                or tokenB.dep_ == 'amod'
                or tokenB.dep_ == 'nummod'
                or tokenB.dep_ == 'compound'
                or tokenB.dep_ == 'case'
                or (tokenB.dep_ == 'punct'
                    and tokenA in tokenB.ancestors)
                or (tokenB.dep_ == 'neg'
                    and getHead(tokenA) is not None
                    and getHead(tokenB).dep_ == 'det')):
            return False
        if (tokenA.dep_ == 'advmod'
            or tokenA.dep_ == 'acomp'
            or tokenA.dep_ == 'prep'
            or (tokenA.dep_ == 'neg'
                and head is not None
                and (head.tag_.startswith('J')
                     or head.tag_.startswith('R')))) \
           and (tokenB.dep_ == 'advmod'):
            return False
    return True


def isLeftEdge(token: Token, sentence: Span):
    """
     This function indicates whether a token is the
     leftmost element in a sentence
    """
    for tok in sentence:
        if tok == token:
            return True
        else:
            break
    return False

def hasLeftChildren(tok: Token):
    """
     This function indicates whether the token input to the
     function has any children to its left
    """
    for child in tok.children:
        if child.i < tok.i:
            return True
    return False


def isRoot(token):
    if token == token.head \
       or token.dep_ == 'ROOT':
        return True
    elif (token.dep_ == 'conj'
          and token.head == token.head.head):
        return True
    else:
        return False


def takesBareInfinitive(item: Token):
    """
     This function exists because spaCY uses the same dependency
     configuration for tensed clauses and untensed clauses (so-called
     "small clauses"). We need to know when something is a small clause
     so we know how to indent the tree properly, among other things.
     The list in this function may not be complete -- the correct
     list should be reviewed.
    """
    if item is None:
        return False
    if item.lemma_ in ["make",
                       "have",
                       "help",
                       "let",
                       "go",
                       "bid",
                       "feel",
                       "hear",
                       "see",
                       "watch",
                       "notice",
                       "observe",
                       "overhear",
                       "monitor",
                       "help",
                       "observe",
                       "perceive",
                       "notice",
                       "consider",
                       "proclaim",
                       "declare"]:
        return True
    return False

def tensed_clause(tok: Token):
    """
     This function calculates whether a token is the head of a tensed clause.
     We need to know if a clause is a tensed clause to count subordinate
     clauses and complement clauses correctly and indent them properly.
     Basically, tensed clauses are either built around the main verb of the
     sentence or they have a subject and are not infinitives or the complements
     of the verbs that take a bare infinitive, e.g., make, have, bid, and let
"""
    # does it have a subject?
    hasSubj = False
    hasTenseMarker = False
    infinitive = False
    head = getHead(tok)
    for child in tok.children:
        # tensed clauses obligatorily contain subjects
        if child.dep_ in ['nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'expl']:
            hasSubj = True
        # infinitives are never tensed clauses
        if child.dep_ == 'aux' and child.tag_ == 'TO':
            hasTenseMarker = False
            infinitive = True
            break
        # a tensed clause has to contain a tensed verb,
        # which may be an auxiliary
        if child.dep_ == 'aux' \
           and (child.tag_ == 'MD'
                or child.lemma_ in ['am',
                                    'are',
                                    'is',
                                    'was',
                                    'were',
                                    'have',
                                    'has',
                                    'do',
                                    'does']
                or child.tag_ == 'BEZ'
                or child.tag_ == 'BEM'
                or child.tag_ == 'BER'
                or child.tag_.startswith('HV')
                or child.tag_.startswith('DO')
                or 'Tense=' in str(child.morph)):
            hasTenseMarker = True

    # if we're at root level, we still have to check if we have
    # a tensed verb, which may be an auxiliary
    if tok.tag_ == 'MD' \
       or tok.tag_ == 'BEZ' \
       or tok.tag_ == 'BEM' \
       or tok.tag_ == 'BER' \
       or tok.tag_.startswith('HV') \
       or tok.tag_.startswith('DO') \
       or 'Tense=' in str(tok.morph):
        hasTenseMarker = True

    if infinitive:
        return False
    # Imperatives count as tensed
    if not hasTenseMarker \
       and not hasSubj \
       and isRoot(tok) \
       and tok.text != tok.lemma_:
        return True
    # Otherwise subjectless verbs count as not tensed
    elif not hasSubj:
        return False
    # Otherwise inflected verbs count as not tensed
    elif (not hasTenseMarker
          and tok.tag_ != 'VBZ'
          and tok.lower_ != tok.lemma_):
        return False
    # Subjects of small clauses (object + bare infinitive)
    # do not count as tensed
    elif (head is not None
          and takesBareInfinitive(head)
          and tok.lower_ == tok.lemma_
          and hasSubj):
        return False
    return True


def getTensedVerbHead(token):
    if isRoot(token):
        return token

    if tensed_clause(token):
        return token
    if token.lower_ == 'be' \
       and 'MD' in [child.tag_ for child in token.children]:
        return token

    if token.morph is not None \
       and 'PunctSide=Ini' in str(token.morph) \
       and isRoot(token):
        if token.i + 1 < len(token.doc) \
           and token.nbor(1) is not None:
            if token.nbor(1) is None:
                return token
            return getTensedVerbHead(token.nbor(1))
    if token.tag_ in ['VBD', 'VBZ', 'MD'] and token.dep_ != 'aux':
        return token
    if token.pos_ == 'VERB':
        if isRoot(token) and 'VerbForm=Inf' in str(token.morph):
            return token
        elif token.dep_ == 'xcomp' and 'VerbForm=Inf' in str(token.morph):
            return token
        elif 'Tense=Past' in str(token.morph):
            return token
        elif 'Tense=Pres' in str(token.morph):
            return token
        elif 'am' in [item.lower_ for item in token.children]:
            return token
        elif 'are' in [item.lower_ for item in token.children]:
            return token
        elif 'was' in [item.lower_ for item in token.children]:
            return token
        elif 'were' in [item.lower_ for item in token.children]:
            return token
        elif 'do' in [item.lower_ for item in token.children]:
            return token
        elif 'does' in [item.lower_ for item in token.children]:
            return token
        elif 'did' in [item.lower_ for item in token.children]:
            return token
        elif 'have' in [item.lower_ for item in token.children]:
            return token
        elif 'has' in [item.lower_ for item in token.children]:
            return token
        elif 'had' in [item.lower_ for item in token.children]:
            return token
        elif 'can' in [item.lower_ for item in token.children]:
            return token
        elif 'could' in [item.lower_ for item in token.children]:
            return token
        elif 'will' in [item.lower_ for item in token.children]:
            return token
        elif 'would' in [item.lower_ for item in token.children]:
            return token
        elif 'should' in [item.lower_ for item in token.children]:
            return token
        elif 'may' in [item.lower_ for item in token.children]:
            return token
        elif 'might' in [item.lower_ for item in token.children]:
            return token
        elif 'must' in [item.lower_ for item in token.children]:
            return token
        elif '\'d' in [item.lower_ for item in token.children]:
            return token
        elif '\'s' in [item.lower_ for item
                       in token.children if item.dep_ == 'aux']:
            return token
        elif '\'ve' in [item.lower_ for item in token.children]:
            return token
        elif '\'ll' in [item.lower_ for item in token.children]:
            return token
        elif '’d' in [item.lower_ for item in token.children]:
            return token
        elif '’s' in [item.lower_ for item
                      in token.children if item.dep_ == 'aux']:
            return token
        elif '’ve' in [item.lower_ for item in token.children]:
            return token
        elif '’ll' in [item.lower_ for item in token.children]:
            return token
        elif (token.dep_ == 'conj'
            or token.tag_ in ['VBG', 'VBN']
            or ('TO' in [item.tag_ for item in token.children])
                and not isRoot(token)):
            if token.head is None:
                return token
            if token.dep_ == 'conj' and tensed_clause(token.head):
                return token
            if tensed_clause(token.head):
                return token.head
            return getTensedVerbHead(token.head)
        else:
            if token.head is None:
                return token
            return getTensedVerbHead(token.head)
    elif isRoot(token):
        return None
    else:
        if token.head is None:
            return token
        return getTensedVerbHead(token.head)

def getRoot(token):
    """
    This function returns the sentence root for the current token.
    """
    if isRoot(token):
        return token
    if token.dep_ == '':
        return token
    if token.head is None:
        return token
    return getRoot(token.head)

def pastThreshold(tok, trait, rule, sign, store, include_stops):
     if tok.lemma_ in store:
         return True
     if tok.pos_ in rule:
         threshold = rule[tok.pos_]
     elif tok.text == '.' and 'PERIOD' in rule:
         threshold = rule['PERIOD']
     elif tok.text == '"' and 'QUOTE' in rule:
         threshold = rule['QUOTE']
     else:
         threshold = rule['DEFAULT']
     if sign == 'above' and getattr(tok._, trait) >= threshold:
         if tok.lemma_ not in store and (tok.lemma_ not in stops or include_stops):
             store.append(tok.lemma_)
         return True
     if sign == 'below' and getattr(tok._, trait) <= threshold:
         if tok.lemma_ not in store and (tok.lemma_ not in stops or include_stops):
             store.append(tok.lemma_)
         return True
     return False

def meanAbove(doc, head, trait, threshold):
    total = 0;
    length = 1 + head.right_edge.i - head.left_edge.i
    for tok in doc[head.left_edge.i:head.right_edge.i+1]:
        total += getattr(tok._,trait)
    if total/length < threshold:
        return False
    else:
        return True

def dimHighlight(doc, head, method):
    for tok in doc[head.left_edge.i:head.right_edge.i+1]:
        if method == 'spread' and getTensedVerbHead(tok) != getTensedVerbHead(head):
            continue
        tok._.dim_highlight = True

def negHighlight(doc, head, trait, rule, method, include_stops):
    for tok in doc[head.left_edge.i:head.right_edge.i+1]:
        if method == 'spread' and getTensedVerbHead(tok) != getTensedVerbHead(head):
            continue
        if rule is None or not pastThreshold(tok, trait, rule, "above", [], include_stops):
            tok._.neg_highlight = True


def apply_highlights(doc, trait, method, rule, color1, color2, negRule, color3, color4, include_stops):
   html_string = ''
   store = []
   for tok in doc:
       if rule is not None:
           pastThreshold(tok, trait, rule, 'above', store, include_stops)
   for tok in doc:
        if tok.lemma_ in store \
           or (rule is not None and tok == getTensedVerbHead(tok) \
            and meanAbove(doc, getTensedVerbHead(tok), trait, rule['DEFAULT'])) \
           or (rule is not None \
            and pastThreshold(tok, trait, rule, 'above', store, include_stops)) \
            and meanAbove(doc, getTensedVerbHead(tok), trait, -0.3):
            if method in ['spread', 'fullspread']:
                dimHighlight(doc, getTensedVerbHead(tok), method)
                tok._.bright_highlight = True
            elif method == 'multilevel':
                if 'SUBRULE' in rule and pastThreshold(tok, trait, rule['SUBRULE'], 'above', [], include_stops):
                    tok._.bright_highlight = True
                else:
                    tok._.dim_highlight = True
            else:   
                tok._.bright_highlight = True
        if negRule is not None \
           and pastThreshold(tok, trait, negRule, 'below', [], include_stops):
            if method in ['spread', 'fullspread']:
                negHighlight(doc, getTensedVerbHead(tok), trait, rule, method, include_stops)
                tok._.bright_neg_highlight = True
            elif method == 'multilevel':
                if 'SUBRULE' in rule and pastThreshold(tok, trait, negRule['SUBRULE'], 'below', [], include_stops):
                    tok._.bright_neg_highlight = True
                else:
                    tok._.neg_highlight = True                
            else:
                tok._.bright_neg_highlight = True
   template = (
               '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
              )
   for tok in doc:
       if tok._.bright_neg_highlight:
           html_string += template.format(color3, tok.text_with_ws)                     
       elif tok._.neg_highlight:
           html_string += template.format(color4, tok.text_with_ws)          
       elif tok._.bright_highlight:
           html_string += template.format(color1, tok.text_with_ws)
       elif tok._.dim_highlight:
           html_string += template.format(color2, tok.text_with_ws)
       else:
           html_string += tok.text_with_ws
   return html_string

def get_html_string(doc, tokens, token_scores, trait, method, rule, color1, color2, negRule, color3, color4, include_stops):
    for token in doc:
        token._.dim_highlight = False
        token._.bright_highlight = False
        token._.neg_highlight = False
        token._.bright_neg_highlight = False
    aligned_tokens, aligned_scores = align(doc, tokens, token_scores, [], [])
    add_trait(doc, trait, aligned_scores)
    return apply_highlights(doc, trait, method, rule, color1, color2, negRule, color3, color4, include_stops)
