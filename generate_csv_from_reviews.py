from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def file_review_to_df(file_path, sentiment=np.nan, category=np.nan):

    #print(file_path)
    df = pd.DataFrame()
    
    #f = open(file_path, "r", encoding="utf8", errors='ignore')
    f = open(file_path, "r")
    
    soup = BeautifulSoup(f, "html.parser")
    reviews = soup.find_all("review")
    
    # for each review we create a dataframe with only one row
    # we're gonna append this dataframe to the main one
    for review in reviews:
        tmp_df_review = pd.DataFrame()
        tmp_child_list = []
        # list childrens of this review
        for child in review.children:
            if child.name != None:
                if child.name not in tmp_child_list:
                    tmp_df_review[child.name] = [review.find(child.name).text.strip("\n")]
                    tmp_child_list.append(child.name)
        df = df.append(tmp_df_review, ignore_index=True)

    df["sentiment"] = sentiment
    df["category"] = category

    return df


def generate_csv_from_reviews(file_path):
	"""
	read reviews in sorted data
	save pos reviews and neg reviews into a DataFrame
	save this DataFrame as csv in file_path
	"""

	# get list of directories containing reviews
	dir_category_list = os.listdir('sorted_data')
	tmp_list = []
	for item in dir_category_list:
	    if os.path.isdir("sorted_data/"+item):
	        tmp_list.append(item)
	dir_category_list = tmp_list
	dir_category_list.pop(0)

	# create dataframe, we're gonna append all reviews in it
	df = pd.DataFrame()

	# for every directory containing data, we're gonna append positive/negative reviews to our dataframe
	for i in tqdm(range(len(dir_category_list))):
	    df = df.append(file_review_to_df("sorted_data/"+dir_category_list[i]+"/positive.review", sentiment=1, category=dir_category_list[i]), ignore_index=True)
	    df = df.append(file_review_to_df("sorted_data/"+dir_category_list[i]+"/negative.review", sentiment=0, category=dir_category_list[i]), ignore_index=True)    

	# we'll keep only review_text, sentiment and rating
	df_reviews = df[["review_text", "sentiment", "rating"]]
	df_reviews["rating"] = df_reviews["rating"].astype(float).astype(int)
	print(df_reviews.info())

	# save the dataframe as csv
	df_reviews.to_csv(file_path, index=False)


if __name__ == "__main__":

	generate_csv_from_reviews("labeled_reviews.csv")
