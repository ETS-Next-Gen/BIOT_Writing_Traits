```
Installation:
pip install -r requirements.txt
python -m spacy download en_core_web_sm

Sample run commands:

To run the default csv file containing multiple essays:
python src/main.py -i tests/input/persuade_2.0_sample.csv -r data/rotation/ -v -o tests/output/persuade.csv

To do the same run but adding flags that make the key fields in the csv file explicit:

python src/main.py -i tests/input/persuade_2.0_sample.csv -r data/rotation/ -v -o tests/output/persuade.csv -d ID -t full_text

to do the same but add a list of fields from the input file that are to be "passed through" and kept in the output file:

python src/main.py -i tests/input/persuade_2.0_sample.csv -r data/rotation/ -v -o tests/output/persuade.csv -s prompt_name,task,assignment,source_text,gender,grade_level,ell_status,race_ethnicity,economically_disadvantaged,student_disability_status -d ID -t full_text

to run a single file and output not only trait data for whole essays but trait information by token in a csv file (-c) and visualizations by token (-p):

python src/main.py -i data/essays/test_essay.txt -r data/rotation/ -o tests/output/out.json -c -p

```
