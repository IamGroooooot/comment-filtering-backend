import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sys
from sklearn.externals import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)
modelPath = 'model.pkl'
smodelPath = 'savedmodel.pkl'
model = pickle.load(open(modelPath, 'rb'))
savedmodel = pickle.load(open(modelPath, 'rb'))
#savedmodel = joblib.load(smodelPath)
print(sys.path)

# 축약어 모음
APPOS = {
    "aren't" : "are not",
    "can't" : "cannot",
    "couldn't" : "could not",
    "didn't" : "did not",
    "doesn't" : "does not",
    "don't" : "do not",
    "hadn't" : "had not",
    "hasn't" : "has not",
    "haven't" : "have not",
    "he'd" : "he would",
    "he'll" : "he will",
    "he's" : "he is",
    "i'd" : "i would",
    "i'd" : "i had",
    "i'll" : "i will",
    "i'm" : "i am",
    "isn't" : "is not",
    "it's" : "it is",
    "it'll":"it will",
    "i've" : "i have",
    "let's" : "let us",
    "mightn't" : "might not",
    "mustn't" : "must not",
    "shan't" : "shall not",
    "she'd" : "she would",
    "she'll" : "she will",
    "she's" : "she is",
    "shouldn't" : "should not",
    "that's" : "that is",
    "there's" : "there is",
    "they'd" : "they would",
    "they'll" : "they will",
    "they're" : "they are",
    "they've" : "they have",
    "we'd" : "we would",
    "we're" : "we are",
    "weren't" : "were not",
    "we've" : "we have",
    "what'll" : "what will",
    "what're" : "what are",
    "what's" : "what is",
    "what've" : "what have",
    "where's" : "where is",
    "who'd" : "who would",
    "who'll" : "who will",
    "who're" : "who are",
    "who's" : "who is",
    "who've" : "who have",
    "won't" : "will not",
    "wouldn't" : "would not",
    "you'd" : "you would",
    "you'll" : "you will",
    "you're" : "you are",
    "you've" : "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll":" will",
    "didn't": "did not"
}
# 불용어 모음
STOPWORDS = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()
tokenizer = TweetTokenizer()
analyzer = SentimentIntensityAnalyzer()

# 여러 분류의 toxic feature를 istoxic으로 변환하는 함수
def cat_istoxic(count):
    if count < 1:
        return 0
    else:
        return 1    
    
def get_istoxic(df):
    df['sum_of_toxic'] = df['toxic'] + df['severe_toxic'] + df['obscene'] + df['threat'] + df['insult'] + df['identity_hate']
    df['is_toxic'] = df['sum_of_toxic'].apply(lambda x: cat_istoxic(x))
    df = df.drop(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'sum_of_toxic'], axis=1)

    return df

# 단어, 문장 등을 count하여 feature로 넣는 함수 
def get_features_count(df):
    df['count_of_sent'] = df['comment_text'].apply(lambda x: len(re.findall('\n', str(x)))+1)
    df['count_of_word'] = df['comment_text'].apply(lambda x: len(str(x).split()))
    df['count_of_unique_word'] = df['comment_text'].apply(lambda x: len(set(str(x).split())))
    df['count_of_punctuations'] = df['comment_text'].apply(lambda x: len([s for s in str(x) if s in string.punctuation]))
    df['count_of_upper_words'] = df['comment_text'].apply(lambda x: len([s for s in str(x).split() if s.isupper()]))
    df['count_of_stopwords'] = df['comment_text'].apply(lambda x: len([s for s in str(x).lower().split() if s in STOPWORDS]))

    df['unique_word_percent'] = df['count_of_unique_word'] * 100 / df['count_of_word']
    df['punct_percent'] = df['count_of_punctuations'] * 100 / df['count_of_word']

    return df

# 쓸모없는 정보(개행, IP주소, USERNAME)을 제거하는 함수
def clean_useless(comment):
    comment = comment.lower()
    
    # 개행 제거
    comment = re.sub('\\n', " ", comment)
    # IP주소 제거
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)
    # USERNAME 제거
    comment = re.sub("\[\[.*\]","",comment)
    
    # 토큰화
    words = tokenizer.tokenize(comment)
    
    # 줄임말 풀이
    words = [APPOS[word] if word in APPOS else word for word in words]
    # 줄임말 다시 분리
    sent = " ".join(words)
    words = tokenizer.tokenize(sent)
    # stemming
    words = [lemmatizer.lemmatize(word, "v") for word in words]
    # 불용어 제거
    words = [w for w in words if not w in STOPWORDS]
    
    
    clean_sent = " ".join(words)
    
    return(clean_sent)

# 위의 함수를 적용하여 comment를 clean하게 만드는 함수
def get_clean_comment(df):
    df['comment_text'] = df['comment_text'].apply(lambda x: clean_useless(x))
    
    return df

# 감정분석을 하는 함수
def vader_polarity(sentence, threshold=0.1):
    scores = analyzer.polarity_scores(sentence)
    
    # compound 값에 기반하여 threshold 입력값보다 크면 1, 그렇지 않으면 0을 반환 
    # compound 가 0.1 보다 크면 긍정 그렇지 않으면 부정
    agg_score = scores['compound']
    final_sentiment = 1 if agg_score >= threshold else 0
    
    return final_sentiment

# 감정분석을 하여 sent_scores라는 피처를 추가하는 함수 
def get_sent_scores(df):
    df['sent_scores'] = df['comment_text'].apply(lambda x: vader_polarity(x))
    
    return df

# 전처리 종합 함수
def preprocessing(df):
    get_istoxic(df)
    get_features_count(df)
    get_clean_comment(df)
    get_sent_scores(df)
    
    df = df.drop(['id', 'comment_text'], axis=1)
    
    return df

@app.route('/')
def home():
    print("Home진입")
    #return "hello"
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    savedmodel.predict()


    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='댓글이 악성일 확률 {}%'.format(output*100))

@app.route('/results', methods=['POST'])
def results():
    print("result 실행 됨------------------")
    print("결과: " + str(request))
    #print("결과: " + str(list(request.get_json(force=True).values())))
    try:
        data = request.get_json(force=True)
    except:
        print("json 파싱 실패")
        return jsonify(-1)
    else:
        print("json 파싱 성공")

    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    print("----------------------")
    return jsonify(output)

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
