from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils.load_datasets import load_MR, load_Semeval2017A
from training import get_metrics_report


DATASET = 'MR'
#PRETRAINED_MODEL = 'siebert/sentiment-roberta-large-english'
#PRETRAINED_MODEL = 'juliensimon/reviews-sentiment-analysis'
PRETRAINED_MODEL = 'Ibrahim-Alam/finetuning-xlnet-base-cased-on-tweet_sentiment_binary'


#DATASET = 'Semeval2017A'
#PRETRAINED_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment'
#PRETRAINED_MODEL = 'hakonmh/sentiment-xdistil-uncased'
#PRETRAINED_MODEL = 'ProsusAI/finbert'

LABELS_MAPPING = {
    'siebert/sentiment-roberta-large-english': {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
    },
    'cardiffnlp/twitter-roberta-base-sentiment': {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
    },
    'hakonmh/sentiment-xdistil-uncased': {
        'Negative': 'negative',
        'Neutral': 'neutral',
        'Positive': 'positive',
    },
    'ProsusAI/finbert': {
        'neutral': 'neutral',
        'positive': 'positive',
        'negative': 'negative',
    },
    'juliensimon/reviews-sentiment-analysis': {
        'LABEL_1': 'positive',
        'LABEL_0': 'negative',
    },
    'Ibrahim-Alam/finetuning-xlnet-base-cased-on-tweet_sentiment_binary': {
        'LABEL_1': 'positive',
        'LABEL_0': 'negative',
    },
}

if __name__ == '__main__':
    # load the raw data
    if DATASET == "Semeval2017A":
        X_train, y_train, X_test, y_test = load_Semeval2017A()
    elif DATASET == "MR":
        X_train, y_train, X_test, y_test = load_MR()
    else:
        raise ValueError("Invalid dataset")

    # encode labels
    le = LabelEncoder()
    le.fit(list(set(y_train)))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    n_classes = len(list(le.classes_))


    # define a proper pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model=PRETRAINED_MODEL)
    #sentiment_pipeline = pipeline(model="roberta-large-mnli")

    y_pred = []

    cnt = 0 
    for x in tqdm(X_test):
        # TODO: Main-lab-Q6 - get the label using the defined pipeline 
        label = sentiment_pipeline(x)   
        y_pred.append(LABELS_MAPPING[PRETRAINED_MODEL][list(label[0].values())[0]])
        # print(label)
        # cnt += 1
        # if (cnt == 30): break 

    y_pred = le.transform(y_pred)
    #print(y_pred)
    print(f'\nDataset: {DATASET}\nPre-Trained model: {PRETRAINED_MODEL}\nTest set evaluation\n{get_metrics_report([y_test], [y_pred])}')



