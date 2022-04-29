from joblib import dump,load
import pandas as pd
from sklearn.model_selection import train_test_split

model = load('saved_model.joblib')

OUTPUT_PATH ='Fit_dataset.csv'

dataset = pd.read_csv(OUTPUT_PATH, encoding = "ISO-8859-1", error_bad_lines=False)

HEADERS = ["field_1st_author","field_2nd_author","author_fname",    "author_midname", "auth_suffix", "author_lname_IDF",
           "affl_email","affl_jaccard", "affl_tfidf",  "affl_softtfidf", "affl_dept_jaccard", "affl_org_jaccard","affl_location_jaccard",
           "coauth_lname_shared",    "coauth_lname_idf",    "coauth_jaccard", "coauth_lname_finitial_jaccard",
           "mesh_shared", "mesh_shared_idf",    "mesh_tree_shared", "mesh_tree_shared_idf",
           "journal_shared_idf", "journal_year", "journal_year_diff",
           "abstract_jaccard",
           "title_jaccard","title_bigram_jaccard", "title_embedding_cosine", "abstract_embedding_cosine", "target"]

target_index = (len(HEADERS))-1

# def split_dataset(dataset, train_percentage, feature_headers, target_header):
#     """
#     Split the dataset with train_percentage
#     :param dataset:
#     :param train_percentage:
#     :param feature_headers:
#     :param target_header:
#     :return: fit_x
#     :return: train_x, test_x, train_y, test_y
#     """
 
#     # Split dataset into train and test dataset
#     train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
#                                                         train_size=train_percentage)
#     return train_x, test_x, train_y, test_y
fit_x = dataset[HEADERS[2:target_index]]
fix_y = model.predict_proba(fit_x)

predicted = []
for i in range(0,len(fix_y)):
    if(fix_y[i][0]>=0.7):
        predicted.append(0)
    else:
        predicted.append(1)

print(predicted)