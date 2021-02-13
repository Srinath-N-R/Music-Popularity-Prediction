import pandas as pd
import numpy as np
import string
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline

reviews_df = reviews_df[~pd.isnull(reviews_df['reviewText'])]
reviews_df.drop_duplicates(subset=['reviewerID', 'asin', 'unixReviewTime'], inplace=True)
reviews_df.drop('Unnamed: 0', axis=1, inplace=True)
reviews_df.reset_index(inplace=True)

reviews_df['helpful_numerator'] = reviews_df['helpful'].apply(lambda x: eval(x)[0])
reviews_df['helpful_denominator'] = reviews_df['helpful'].apply(lambda x: eval(x)[1])
reviews_df['helpful%'] = np.where(reviews_df['helpful_denominator'] > 0,
                                  reviews_df['helpful_numerator'] / reviews_df['helpful_denominator'], -1)

reviews_df['helpfulness_range'] = pd.cut(x=reviews_df['helpful%'], bins=[-1, 0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                         labels=['empty', '1', '2', '3', '4', '5'], include_lowest=True)

def text_process(reviewText):
    nopunc = [i for i in reviewText if i not in string.punctuation]
    nopunc = nopunc.lower()
    nopunc_text = ''.join(nopunc)
    return [i for i in nopunc_text.split() if i not in stopwords.words('english')]

pipeline = Pipeline([('Tf-Idf', TfidfVectorizer(ngram_range=(1,2), analyzer=text_process)),('classifier', MultinomialNB())])
X = reviews_df['reviewText']
y = reviews_df['helpfulness_range']
review_train, review_test, label_train, label_test = train_test_split(X, y, test_size=0.5)
pipeline.fit(review_train, label_train)
pip_pred = pipeline.predict(review_test)
print(metrics.classification_report(label_test, pip_pred))

rev_test_pred_NB_df = pd.DataFrame(data={'review_test': review_test2, 'prediction': pip_pred2})
rev_test_pred_NB_df.to_csv('rev_test_pred_NB_df.csv')

temp_df = pd.DataFrame(np.unique(reviewers_rating_df['reviewerID']), columns=['unique_ID'])
temp_df['unique_asin'] = pd.Series(np.unique(reviewers_rating_df['asin']))
temp_df['unique_ID_int'] = range(20000, 35998)
temp_df['unique_asin_int'] = range(1, 15999)
reviewers_rating_df = pd.merge(reviewers_rating_df, temp_df.drop(['unique_asin', 'unique_asin_int'], axis=1), left_on='reviewerID', right_on='unique_ID')
reviewers_rating_df = pd.merge(reviewers_rating_df, temp_df.drop(['unique_ID', 'unique_ID_int'], axis=1),left_on='asin', right_on='unique_asin')
reviewers_rating_df['overall_rating'] = reviewers_rating_df['overall']
id_asin_helpfulness_df = reviewers_rating_df[['reviewerID', 'unique_ID_int', 'helpfulness_range']].copy()
# Delete the not in use columns:
reviewers_rating_df.drop(['asin', 'unique_asin', 'reviewerID', 'unique_ID', 'overall', 'helpfulness_range'], axis=1, inplace=True)

matrix = reviewers_rating_df.pivot(index='unique_ID_int', columns='unique_asin_int', values='overall_rating')
matrix = matrix.fillna(0)
user_item_matrix = sparse.csr_matrix(matrix.values)

model_knn = neighbors.NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
model_knn.fit(user_item_matrix)

neighbors = np.asarray(model_knn.kneighbors(user_item_matrix, return_distance=False))

unique_id = []
k_neigh = []
for i in range(15998):
    unique_id.append(i + 20000)
    k_neigh.append(list(neighbors[i][1:10])) #Grabbing the ten closest neighbors
neighbors_df = pd.DataFrame(data={'unique_ID_int': unique_id,
                                  'k_neigh': k_neigh})
id_asin_helpfulness_df = pd.merge(id_asin_helpfulness_df, neighbors_df, on='unique_ID_int')
id_asin_helpfulness_df['neigh_based_helpf'] = id_asin_helpfulness_df['unique_ID_int']

for index, row in id_asin_helpfulness_df.iterrows():
    row = row['k_neigh']
    lista = []
    for i in row:
        p = id_asin_helpfulness_df.loc[i]['helpfulness_range']
        lista.append(p)
    id_asin_helpfulness_df.loc[index, 'neigh_based_helpf'] = np.nanmean(lista)