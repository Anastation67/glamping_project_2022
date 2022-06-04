import typer
import tensorflow as tf
import numpy as np
import re
from tensorflow import _keras_package
from keras.layers import Dense, LSTM, Input, Dropout, Embedding
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
import numpy as np
import chardet
import pandas as pd
import re
import string
import nltk
import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

app = typer.Typer()


def preprocessing(review):

    stop = stopwords.words('russian')
    stop.remove('хорошо')
    words = ['это', 'место', 'персонал', 'очень', 'номер', 'туалет', 'ребенок',
             'весь', 'привет', 'делать', 'конец', 'завтрак', 'опыт',
             'отдых', 'мысль', 'никто', 'почему', 'название', 'хотя',
             'поездка', 'массажист', 'заранее', 'тело', 'пример', 'сформировать']
    for w in words:
        stop.append(w)

    review = review.lower()
    review = re.sub(r'((www.\.\[^\s]+)|(https?://[^\s]+))', r'', review)
    review = re.sub(r'@[^\s]+', r'', review)
    review = re.sub(r'\W*\b\w{1,3}\b', r'', review)

    token = nltk.word_tokenize(review)  # splitting review on tokens

    review = [word for word in token if (word not in stop and word not in string.punctuation
                                         and word != "' '" and word != "``"
                                         and word != "''" and word.isnumeric() == False)]

    review = ' '.join(pymorphy2.MorphAnalyzer().parse(word)[0].normal_form for word in review)  # lemmatization
    return review


def prepareOnePlaceReviews(reviews):
    for i in range(len(reviews)):
        reviews[i] = preprocessing(reviews[i])
    return reviews

def res_to_string(result):
    count_all_positive = 0
    count_all_negative = 0
    if (result[0][0] < result[0][1]):
        count_all_negative += 1;
    elif (result[0][0]) > (result[0][1]):
        count_all_positive += 1;
    return [count_all_positive,count_all_negative]

def cleanData():

    with open('tripadvisor_hotel_reviews.csv', 'rb') as rawdata:
        result = chardet.detect(rawdata.read(10000))

    df = pd.read_csv('tripadvisor_hotel_reviews.csv', encoding=result.get('encoding'))
    # df.head

    # rename column
    df.rename(columns={
        'selection1_name,"selection1_selection2_name","selection1_selection2_selection3","selection1_selection2_selection3_url"': 'reviews'},
        inplace=True)

    # splitting into two columns
    places = []
    reviews = []
    for element in range(df.size):
        place = df['reviews'][element].split('"')[0]
        review = df['reviews'][element].split('"')[1]
        places.append(place[:-1])
        reviews.append(review)

    df = pd.DataFrame({'places': places, 'reviews': reviews})  # dataframe with already splitted data

    # collecting reviews of each place
    df1 = df.copy()
    places1 = []
    data = {}

    for row in range(len(df1.index)):
        if df1.iat[row, 0] not in places1:
            places1.append(df1.iat[row, 0])
            df2 = df1[df1['places'] == df1.iat[row, 0]]
            tem = []
            for i in range(len(df2.index)):
                tem.append(df2.iat[i, 1].lower())
            data[df1.iat[row, 0]] = tem

    reviews1 = [] * len(places1)

    for i in range(len(places1)):
        reviews1.append(data[places1[i]])
    #     reviews1[i] = ' '.join(reviews1[i])

    new = pd.DataFrame({'places': places1, 'collected_reviews': reviews1})
    # new # dataframe is now ready to be cleaned

    new['collected_reviews'] = new['collected_reviews'].apply(lambda reviews: prepareOnePlaceReviews(reviews))

    return new


def showTonality(new):
    f = open('train_data_true.txt', 'r', encoding='utf-8')
    texts_true = f.readlines()
    texts_true[0] = texts_true[0].replace('\ufeff', '')

    f = open('train_data_false.txt', 'r', encoding='utf-8')
    texts_false = f.readlines()
    texts_false[0] = texts_false[0].replace('\ufeff', '')

    texts = texts_true + texts_false
    count_true = len(texts_true)
    count_false = len(texts_false)
    total_lines = count_true + count_false
    # print(count_true, count_false, total_lines)

    maxWordsCount = 2000
    tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', lower=True,
                          split=' ', char_level=False)
    tokenizer.fit_on_texts(texts)

    data = tokenizer.texts_to_sequences(texts)
    max_text_len = 10
    data_pad = tf.keras.utils.pad_sequences(data, maxlen=max_text_len)
    # print(data_pad)
    # print(data_pad.shape)

    X = data_pad
    Y = np.array([[1, 0]] * count_true + [[0, 1]] * count_false)
    # print(X.shape, Y.shape)

    indeces = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
    X = X[indeces]
    Y = Y[indeces]

    model = Sequential()
    model.add(Embedding(maxWordsCount, 128, input_length=max_text_len))
    model.add(LSTM(64, activation='tanh', return_sequences=True))
    model.add(LSTM(32, activation='tanh'))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(0.0001))

    history = model.fit(X, Y, batch_size=32, epochs=50)

    # predicting model
    tup_pos = []
    tup_neg = []
    tup = []

    for i in range(new.shape[0]):
        a = 0
        b = 0
        tup_tonality = [] * len(new.iat[i, 1])
        for j in range(len(new.iat[i, 1])):
            t = new.iat[i, 1][j]
            data = tokenizer.texts_to_sequences([t])
            data_pad = tf.keras.utils.pad_sequences(data, maxlen=max_text_len)
            res = model.predict(data_pad)
            tup_tonality.append(res_to_string(res))
            # print(res)
            # print(res_to_string(res))
            a += res_to_string(res)[0]
            b += res_to_string(res)[1]
        tup_pos.append(a)
        tup_neg.append(b)
        tup.append(tup_tonality)

    return tup_pos, tup_neg, tup

@app.command()
def start():
    # cleaning data
    new = cleanData()

    # learning Neural Network &  predicting model
    tup_pos, tup_neg, tup = showTonality(new)

    # making it beautiful
    new['Number of positive reviews'] = tup_pos
    new['Number of negative reviews'] = tup_neg
    for i in range(new.shape[0]):
        if new.iat[i, 1] == ['']:
            new.iat[i, 3] = new.iat[i, 2]

    new['Total number of reviews'] = new['Number of positive reviews'] + new['Number of negative reviews']

    conditions = [(new['Number of positive reviews'] > 1.25 * new['Number of negative reviews']),
                  (new['Number of positive reviews'] < 1.25 * new['Number of negative reviews'])
                  & (new['Number of positive reviews'] > new['Number of negative reviews']),
                  (new['Number of positive reviews'] != 0) & (new['Number of negative reviews'] != 0)
                  & (new['Number of positive reviews'] == new['Number of negative reviews']),
                  (new['Number of positive reviews'] < new['Number of negative reviews'])
                  & (1.25 * new['Number of positive reviews'] > new['Number of negative reviews']),
                  (1.25 * new['Number of positive reviews'] < new['Number of negative reviews']),
                  (new['Number of positive reviews'] == 0) & (new['Number of negative reviews'] == 0)]
    five, four, three, two, one, zero = 5, 4, 3, 2, 1, 0
    output = [five, four, three, two, one, zero]
    new['Rating'] = np.select(conditions, output)
    new_sorted = new.sort_values(by=['Rating'], ascending=False)
    new_sorted = new_sorted.drop(columns=['collected_reviews'])

    print(new_sorted)
    print('The data is ready:)\n'
          'Please enter "details place" in case you want to get the analysis of the glamping place\n'
          'Please, enter the place without whitespaces!')

@app.command()
def details(place: str):

    new_temp = cleanData()
    tup = showTonality(new_temp)[2]
    new_temp['tonality_of_each_review'] = tup
    for i in range(new_temp.shape[0]):
        new_temp['places'][i] = new_temp['places'][i].replace(' ', '')

    positive = []
    negative = []

    for p in range(len(new_temp.index)):
        pos_reviews = []
        neg_reviews = []
        for k in range(len(new_temp.iat[p, 2])):
            if new_temp.iat[p, 2][k] == [1, 0]:
                pos_reviews.append(new_temp.iat[p, 1][k])
            else:
                neg_reviews.append(new_temp.iat[p, 1][k])
        positive.append(pos_reviews)
        negative.append(neg_reviews)

    new_temp['positive_reviews'] = positive
    new_temp['negative_reviews'] = negative

    ready_for_clouds = new_temp[['places', 'positive_reviews', 'negative_reviews']]

    # drawing clouds
    stop = stopwords.words('russian')
    stop.remove('хорошо')
    words = ['это', 'место', 'персонал', 'очень', 'номер', 'туалет', 'ребенок',
             'весь', 'привет', 'делать', 'конец', 'завтрак', 'опыт',
             'отдых', 'мысль', 'никто', 'почему', 'название', 'хотя',
             'поездка', 'массажист', 'заранее', 'тело', 'пример', 'сформировать']
    for w in words:
        stop.append(w)

    # positive clouds
    words_list = ready_for_clouds[ready_for_clouds['places'] == place]['positive_reviews'].tolist()[0]
    pos_words = ' '.join(words_list)

    if pos_words != '':
        pos_wordcloud = WordCloud(background_color='white', width=800, height=500, stopwords=stop).generate(pos_words)
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(pos_wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()

    # negative clouds
    words_list = ready_for_clouds[ready_for_clouds['places'] == place]['negative_reviews'].tolist()[0]
    neg_words = ' '.join(words_list)

    if neg_words != '':
        neg_wordcloud = WordCloud(background_color='white', width=800, height=500, stopwords=stop).generate(neg_words)
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(neg_wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()

    print('Hope you have enjoyed the visit.\n'
          'See you soon :) ')


if __name__ == "__main__":
    app()
    input('Press ENTER to exit')