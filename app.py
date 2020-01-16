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
smodelPath = 'savedmodel.pkl'
savedmodel = joblib.load(smodelPath)
tfidf_vect = joblib.load('tfidf.pkl')

''' nltk.txt에 명시하면 자동으로 다운함
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')
'''

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

# 욕 모음
BADWORDS = ['2g1c',
 '2 girls 1 cup',
 'acrotomophilia',
 'alabama hot pocket',
 'alaskan pipeline',
 'anal',
 'anilingus',
 'anus',
 'apeshit',
 'arsehole',
 'ass',
 'asshole',
 'assmunch',
 'auto erotic',
 'autoerotic',
 'babeland',
 'baby batter',
 'baby juice',
 'ball gag',
 'ball gravy',
 'ball kicking',
 'ball licking',
 'ball sack',
 'ball sucking',
 'bangbros',
 'bareback',
 'barely legal',
 'barenaked',
 'bastard',
 'bastardo',
 'bastinado',
 'bbw',
 'bdsm',
 'beaner',
 'beaners',
 'beaver cleaver',
 'beaver lips',
 'bestiality',
 'big black',
 'big breasts',
 'big knockers',
 'big tits',
 'bimbos',
 'birdlock',
 'bitch',
 'bitches',
 'black cock',
 'blonde action',
 'blonde on blonde action',
 'blowjob',
 'blow job',
 'blow your load',
 'blue waffle',
 'blumpkin',
 'bollocks',
 'bondage',
 'boner',
 'boob',
 'boobs',
 'booty call',
 'brown showers',
 'brunette action',
 'bukkake',
 'bulldyke',
 'bullet vibe',
 'bullshit',
 'bung hole',
 'bunghole',
 'busty',
 'butt',
 'buttcheeks',
 'butthole',
 'camel toe',
 'camgirl',
 'camslut',
 'camwhore',
 'carpet muncher',
 'carpetmuncher',
 'chocolate rosebuds',
 'circlejerk',
 'cleveland steamer',
 'clit',
 'clitoris',
 'clover clamps',
 'clusterfuck',
 'cock',
 'cocks',
 'coprolagnia',
 'coprophilia',
 'cornhole',
 'coon',
 'coons',
 'creampie',
 'cum',
 'cumming',
 'cunnilingus',
 'cunt',
 'dafuq',
 'dank',
 'darkie',
 'date rape',
 'daterape',
 'deep throat',
 'deepthroat',
 'dendrophilia',
 'dick',
 'dork',
 'dildo',
 'dingleberry',
 'dingleberries',
 'dips hit',
 'dirty pillows',
 'dirty sanchez',
 'doggie style',
 'doggiestyle',
 'doggy style',
 'doggystyle',
 'dog style',
 'dolcett',
 'domination',
 'dominatrix',
 'dommes',
 'donkey punch',
 'double dong',
 'double penetration',
 'douche',
 'douchebag',
 'dumbass',
 'dp action',
 'dry hump',
 'dvda',
 'eat my ass',
 'ecchi',
 'ejaculation',
 'erotic',
 'erotism',
 'escort',
 'eunuch',
 'fag',
 'faggot',
 'fecal',
 'felch',
 'fellatio',
 'feltch',
 'female squirting',
 'femdom',
 'figging',
 'fingerbang',
 'fingering',
 'fisting',
 'foot fetish',
 'footjob',
 'frotting',
 'fuck',
 'fuck buttons',
 'fuckin',
 'fucking',
 'fucktards',
 'fudge packer',
 'fudgepacker',
 'futanari',
 'gang bang',
 'gay sex',
 'genitals',
 'giant cock',
 'girl on',
 'girl on top',
 'girls gone wild',
 'goatcx',
 'goatse',
 'god damn',
 'gokkun',
 'golden shower',
 'goodpoop',
 'goo girl',
 'goregasm',
 'grope',
 'group sex',
 'g-spot',
 'guro',
 'hand job',
 'handjob',
 'hard core',
 'hardcore',
 'hentai',
 'hoe',
 'homoerotic',
 'honkey',
 'hooker',
 'hot carl',
 'hot chick',
 'how to kill',
 'how to murder',
 'huge fat',
 'humping',
 'incest',
 'intercourse',
 'jack off',
 'jail bait',
 'jailbait',
 'jelly donut',
 'jerk off',
 'jigaboo',
 'jiggaboo',
 'jiggerboo',
 'jizz',
 'juggs',
 'kike',
 'kinbaku',
 'kinkster',
 'kinky',
 'knobbing',
 'leather restraint',
 'leather straight jacket',
 'lemon party',
 'lolita',
 'lovemaking',
 'make me come',
 'male squirting',
 'masturbate',
 'menage a trois',
 'milf',
 'missionary position',
 'motherfucker',
 'mound of venus',
 'mr hands',
 'muff diver',
 'muffdiving',
 'nambla',
 'nawashi',
 'negro',
 'neonazi',
 'nigga',
 'nigger',
 'nig nog',
 'nimphomania',
 'nipple',
 'nipples',
 'nsfw images',
 'nude',
 'nudity',
 'nympho',
 'nymphomania',
 'octopussy',
 'omorashi',
 'one cup two girls',
 'one guy one jar',
 'orgasm',
 'orgy',
 'paedophile',
 'paki',
 'panties',
 'panty',
 'pedobear',
 'pedophile',
 'pegging',
 'penis',
 'phone sex',
 'piece of shit',
 'pissing',
 'piss pig',
 'pisspig',
 'playboy',
 'pleasure chest',
 'pole smoker',
 'ponyplay',
 'poof',
 'poon',
 'poontang',
 'punany',
 'poop chute',
 'poopchute',
 'porn',
 'porno',
 'pornography',
 'prince albert piercing',
 'pthc',
 'pubes',
 'pussy',
 'queaf',
 'queef',
 'quim',
 'raghead',
 'raging boner',
 'rape',
 'raping',
 'rapist',
 'rectum',
 'reverse cowgirl',
 'rimjob',
 'rimming',
 'rosy palm',
 'rosy palm and her 5 sisters',
 'rusty trombone',
 'sadism',
 'santorum',
 'scat',
 'schlong',
 'scissoring',
 'semen',
 'sex',
 'sexo',
 'sexy',
 'shaved beaver',
 'shaved pussy',
 'shemale',
 'shibari',
 'shit',
 'shitblimp',
 'shitty',
 'shota',
 'shrimping',
 'skeet',
 'slanteye',
 'slut',
 's&m',
 'smut',
 'snatch',
 'snowballing',
 'sodomize',
 'sodomy',
 'spic',
 'splooge',
 'splooge moose',
 'spooge',
 'spread legs',
 'spunk',
 'strap on',
 'strapon',
 'strappado',
 'strip club',
 'style doggy',
 'suck',
 'sucks',
 'suicide girls',
 'sultry women',
 'swastika',
 'swinger',
 'tainted love',
 'taste my',
 'tea bagging',
 'threesome',
 'throating',
 'tied up',
 'tight white',
 'tit',
 'tits',
 'titties',
 'titty',
 'tongue in a',
 'topless',
 'tosser',
 'towelhead',
 'tranny',
 'tribadism',
 'tub girl',
 'tubgirl',
 'tushy',
 'twat',
 'twink',
 'twinkie',
 'two girls one cup',
 'undressing',
 'upskirt',
 'urethra play',
 'urophilia',
 'vagina',
 'venus mound',
 'vibrator',
 'violet wand',
 'vorarephilia',
 'voyeur',
 'vulva',
 'wank',
 'wetback',
 'wet dream',
 'whore',
 'white power',
 'wrapping men',
 'wrinkled starfish',
 'xx',
 'xxx',
 'yaoi',
 'yellow showers',
 'yiffy',
 'zoophilia',
 '__'
 ]

lemmatizer = WordNetLemmatizer()
tokenizer = TweetTokenizer()
analyzer = SentimentIntensityAnalyzer()

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

def preprocessing(comment):
    test_df = pd.DataFrame({'comment_text':[clean_useless(comment)]})
    
    # bad word counting
    test_df['count_of_word'] = len(comment.split())
    test_df['bad'] = sum((1 for word in BADWORDS if comment.count(word)>0))
    test_df['bad_count'] = test_df['bad'] /  test_df['count_of_word']
    
    # sentimental analysis
    sent_score = SentimentIntensityAnalyzer().polarity_scores(comment)
    test_df['sent_scores'] = sent_score['compound']

    # 최종 지표
    test_df['final_scores'] = - test_df['sent_scores'] + test_df['bad_count']
    
    # 불필요한 단어 드롭
    test_df.drop(['count_of_word', 'bad', 'bad_count', 'sent_scores'], axis=1)

    return tfidf_vect.transform(test_df['comment_text']), test_df['final_scores']


# server

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    requestedValues = list(request.form.values())
    # 1 악플, 0 안플아님
    isToxic = savedmodel.predict(preprocessing(requestedValues[1])[0])[0]
    swearing_power = float(preprocessing(requestedValues[1])[1][0])+1
    isToxicText = ""
    if isToxic == 1:
        isToxicText = "악플입니다."
    else:
        isToxicText = "악플이 아닙니다."
        swearing_power = 0

    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)
    #output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text='댓글이 얼마나 악성인가요? {0:.2f}'.format(swearing_power*100.0/3.6), prediction_result = '이 댓글은 '+isToxicText)

@app.route('/results', methods=['POST'])
def results():
    print(">>>>> POST를 명령 받았습니당!!")
    #print("결과: " + str(list(request.get_json(force=True).values())))
    try:
        data = request.get_json(force=True)
    except:
        print("json 파싱 실패")
        return jsonify(-1)
    else:
        print("파싱 성공")

    requestedValues = list(data.values())
    print(">>>>> comment: " + requestedValues[1])
    isToxic = savedmodel.predict(preprocessing(requestedValues[1])[0])[0]
    swearing_power = float(preprocessing(requestedValues[1])[1][0])+1

    output = 0

    if(swearing_power < 0.0):
        swearing_power = 0.0

    if isToxic == 1: # 1이면 악플
        output = 1*10+swearing_power
    else: # 2이면 악플 아님
        output = 2*10+swearing_power
    #prediction = model.predict([np.array(list(data.values()))])
    #output = prediction[0]
    print(">>>>> ----------------------")
    return jsonify(output)

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
