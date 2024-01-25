```
This library supports a method for creating interpretable 
dimensions of an LLM embedding, using a method that rotates
the embedding to maximize explainability with respect to a
list of predictor variables.

The method in question is BIOT 
(paper at https://www.sciencedirect.com/science/article/abs/pii/S0925231221006469, 
github repository at https://github.com/rebeccamarion/BIOT, 
fork for this project at https://github.com/ETS-Next-Gen/BIOT).

The code makes the following assumptions:

1. you start with a csv file that contains 
    (a) an ID field
    (b) a text field containing an essay or other documemt
    (c) one or more predictor variables to be used to align the LLM
    (d) possibly other variables of interest.

2. You run the main module in EMBEDDING MODE to produce the datafiles 
   that are needed to run BIOT

3. You run the BIOT code separately to produce rotation matrices.
   This gives you not only the rotation matrices but weights and
   correlation matrices telling you which dimensions in which hidden
   layers are potentially interpretable. 

4. You run main module in TRAIT SUMMARY MODE to produce trait scores 
   for each essay by applying selected rotation matrices to selected 
   hidden layers. This uses files named trait_vector.json and 
   excluded.json that provide a rule base. These files need to be
   manually created based on decisions about which information from
   which layer to use to create trait scores.

5.Alternatively, you run the main module in TOKEN MODE to produce token
  trait scores and/or highlighting for a specific trait on a specific
  document. The token scores basically tell you which parts of the essay
  contributed to high or low overall scores on a specific trait.

Installation:
pip install -r requirements.txt
python -m spacy download en_core_web_sm

Sample run commands:

EMBEDDING MODE
To run the default csv file to produce the embedding and predictor files needed to run BIOT rotation:

python src/main.py -i tests/input/persuade_2.0_sample.csv -r data/rotation/ -o tests/output/persuade.csv -s holistic_essay_score,VocabularyLength,Organization,VocabularyFrequency,Interactivity,SentenceLength,SentenceStructure,SentenceComplexity,GrammarUsage,Narrativity,Contextualization,Conventionality,Mechanics,Dialogue,Cohesion,Concreteness,StanceTaking -v -e -n

TRAIT SUMMARY MODE

To run the default csv file containing multiple essays:

python src/main.py -i tests/input/persuade_2.0_sample.csv -r data/rotation/ -v -o tests/output/persuade.csv

To do the same run but adding flags that make the key fields in the csv file explicit:

python src/main.py -i tests/input/persuade_2.0_sample.csv -r data/rotation/ -v -o tests/output/persuade.csv -d ID -t full_text

to do the same but add a list of fields from the input file that are to be "passed through" and kept in the output file:

python src/main.py -i tests/input/persuade_2.0_sample.csv -r data/rotation/ -v -o tests/output/persuade.csv -s prompt_name,task,assignment,source_text,gender,grade_level,ell_status,race_ethnicity,economically_disadvantaged,student_disability_status,VocabularyLength,Organization,VocabularyFrequency,Interactivity,SentenceLength,SentenceStructure,SentenceComplexity,GrammarUsage,Narrativity,Contextualization,Conventionality,Mechanics,Dialogue,Cohesion,Concreteness,StanceTaking -d ID -t full_text

TOKEN MODE:

to run a single file and output not only trait data for whole essays but trait information by token in a csv file (-c) and visualizations by token (-p):

python src/main.py -i data/essays/test_essay.txt -r data/rotation/ -o tests/output/out.json -c -p

```
