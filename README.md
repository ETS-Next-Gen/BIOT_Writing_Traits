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

python src/main.py -i tests/input/persuade_2.0_sample.csv -r data/rotation/multigenre/ -o tests/output/persuade.csv -s holistic_essay_score,Formality,VocabularyLength,Organization,VocabularyFrequency,Interactivity,SentenceLength,SentenceStructure,SentenceComplexity,GrammarUsage,Narrativity,Contextualization,Conventionality,Mechanics,Dialogue,Cohesion,Concreteness,StanceTaking -v -e -n

TRAIT SUMMARY MODE

To run the default csv file containing multiple essays:

python src/main.py -i tests/input/persuade_2.0_sample.csv -r data/rotation/persuasive/ -v -o tests/output/persuade.csv

To do the same run but adding flags that make the key fields in the csv file explicit:

python src/main.py -i tests/input/persuade_2.0_sample.csv -r data/rotation/multigenre/ -v -o tests/output/persuade.csv -d ID -t full_text

to do the same but add a list of fields from the input file that are to be "passed through" and kept in the output file:

python src/main.py -i tests/input/persuade_2.0_sample.csv -r data/rotation/multigenre -v -o tests/output/persuade.csv -s full_text,holistic_essay_score,word_count,prompt_name,task	assignment,source_text,gender,grade_level,ell_status,race_ethnicity,economically_disadvantaged,student_disability_status,types.tokens,logwords,dis_coh,Coh_Inf,Coh_Lit,logdta,RefScore3,concrete_5,image_5,concrete_6,concrete_score,Avg_TASA_SFI,image_6,image_score,Verb_Fict_Norm,Mean_ETS_Lexile_Wfreq,nwf_median,Past_Tense_Verb_Norm,Avg_Count_Dep_Clauses,Avg_Wrd_Cnt_Before_Main_Verb,Avg_Wrd_Syll,complex_clauses_norm,Nar_Com_Verb_Norm,Past_Prfct_Aspect_Vrb_Cnt_Norm,Verbs_Conversation_plus2,ZScore1Mean,nsqm,colprep,nsqg,nsqu,Abstract_Nouns_Less127Plus4_Type_Collapsed,Adj_Topical_Norm,Avg_Yngve_Depth,claims_norm,Cog_Prcs_Prcpt_Noun_Norm,Contractions,Coxhead_combined,Nominalization_Lee_Type_Collapsed,svf,verb_choice_norm,WordsInsideQuotes,Academic_Word_List_Ratio,Long_Words,RefScore1,lt_final_scr,Subord_Conces,Negate2,logdtu,nsqs,wordln_2,Avg_Sent_Wrd_Cnt,Prph_Wrd_Cnt_Lngst,dis_coh_length,dis_coh_lexchain,grammaticality,log_types.tokens,Formality,VocabularyLength,Organization,VocabularyFrequency,Interactivity,SentenceLength,SentenceStructure,SentenceComplexity,GrammarUsage,Narrativity,Contextualization,Conventionality,Mechanics,Dialogue,Cohesion,Concreteness,StanceTaking -d essay_id -t essay_text

TOKEN MODE:

to run a single file and output not only trait data for whole essays but trait information by token in a csv file (-c) and visualizations by token (-p):

python src/main.py -i data/essays/test_essay.txt -r data/rotation/multigenre/ -o tests/output/out.json -c -p

```
