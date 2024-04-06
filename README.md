# NLP-GENIUS

## Title to Lyrics Transformer

### Getting the weights

#### Use the pre-computed weights

- A long training has already been made over 1M lyrics (RTX 4090 for 18 hours) :
- Download the weights [here](https://pilou.org/NLP/trained_model.zip).
- Extract the archive in the transformer directory. Your file tree should look like this :

```bash
/NLP-GENIUS$ tree -L 2
.
├── README.md
├── dataset
│   └── song_lyrics.csv
 ...

└── transformers
    ├── Dockerfile
    ├── __pycache__
    ├── classes.py
    ├── preprocessing.py
    ├── requirements.txt
    ├── title_to_lyrics_transformer.py
    ├── trained_model # <- Extracted archive
    └── training.py
```

#### .. Or train from scratch

Configure a .env file in the transformer directory. The following config will train the model for 10 epochs over 1000 rows :

```bash
# /NLP_GENIUS/transformers/.env
TSF_TRAINING=True
TSF_EPOCHS=10
TSF_ROWS=1000
```

Then install the dependencies and run the python script :

```bash
/NLP-GENIUS/transformers$ pip install -r requirements.txt
/NLP-GENIUS/transformers$ python title_to_lyrics_transformer.py
```

### Generating lyrics

#### Build the Docker image

Make sure to be at the root of the project.
First image build will take some time (copying the whole dataset) ~ 12mn.

```bash
 pwd # /NLP-GENIUS

 docker build -t nlp_transformer -f transformers/Dockerfile .
```

#### Run the Docker container

```bash
docker run -e TSF_TITLE="Hello darkness" -it nlp_transformer
```
