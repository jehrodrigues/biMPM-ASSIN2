# biMPM-semantic-textual-similarity
Implementation of Semantic textual similarity with Bilateral Multi-Perspective Matching (biMPM)

### Contents

* [Installation](#installation)
* [Usage](#usage)
  * [Train](#train)

---

## Installation
```
virtualenv venv -p python3
source venv/bin/activate
pip install -r requirements.txt
```
Download SpaCy model:
```
python -m spacy download pt
```

## Usage

### Train biMPM

```
python main.py --model bimpm --dataset assin
```
