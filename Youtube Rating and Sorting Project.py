# Youtube US Videos Measurement Problems Project
# Context
# YouTube (the world-famous video sharing website) maintains a list of the top trending videos on the platform. According to Variety magazine, “To determine the year’s top-trending videos, YouTube uses a combination of factors including measuring users interactions (number of views, shares, comments and likes). Note that they’re not the most-viewed videos overall for the calendar year”. Top performers on the YouTube trending list are music videos (such as the famously virile “Gangam Style”), celebrity and/or reality TV performances, and the random dude-with-a-camera viral videos that YouTube is well-known for.
#
# This dataset is a daily record of the top trending YouTube videos.
#
# Note that this dataset is a structurally improved version of this dataset.
#
# Content
# This dataset includes several months (and counting) of data on daily trending YouTube videos. Data is included for the US, GB, DE, CA, and FR regions (USA, Great Britain, Germany, Canada, and France, respectively), with up to 200 listed trending videos per day.
#
# EDIT: Now includes data from RU, MX, KR, JP and IN regions (Russia, Mexico, South Korea, Japan and India respectively) over the same time period.
#
# Each region’s data is in a separate file. Data includes the video title, channel title, publish time, tags, views, likes and dislikes, description, and comment count.
#
# The data also includes a category_id field, which varies between regions. To retrieve the categories for a specific video, find it in the associated JSON. One such file is included for each of the five regions in the dataset.
#
# For more information on specific columns in the dataset refer to the column metadata.
#
# Acknowledgements
# This dataset was collected using the YouTube API.
#
# Inspiration
# Possible uses for this dataset could include:
#
# Sentiment analysis in a variety of forms
# Categorising YouTube videos based on their comments and statistics.
# Training ML algorithms like RNNs to generate their own YouTube comments.
# Analysing what factors affect how popular a YouTube video will be.
# Statistical analysis over time.
# For further inspiration, see the kernels on this dataset

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################
df_ = pd.read_csv("Data/USvideos.csv")
df = df_.copy()


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)
df.dropna(inplace=True)
df.columns
df = df.drop(
    labels=["trending_date", "title", "channel_title", "category_id", "tags", "thumbnail_link", "comments_disabled",
            "ratings_disabled", "video_error_or_removed", "description"], axis=1)
df.head()


# We dont have a rating score on youtube. So, We can use wilson_lower_bound score for analysis.

def score_like_dislike_diff(like, dislike):
    return like - dislike


df["l-d_score"] = score_like_dislike_diff(df["likes"], df["dislikes"])
df.head()


def score_average_rating(likes, dislikes):
    if likes + dislikes == 0:
        return 0
    return likes / (likes + dislikes)


df["likes_dislikes_average"] = df.apply(lambda x: score_average_rating(x["likes"], x["dislikes"]), axis=1)

df.head()


def wilson_lower_bound(likes, dislikes, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = likes + dislikes
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * likes / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["likes"],
                                                                 x["dislikes"]), axis=1)


df.head()

df.sort_values("wilson_lower_bound", ascending=False).head()
df.head()

# Create a rating score

df["scaled_like"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["likes"]]). \
    transform(df[["likes"]])

df["scaled_dislike"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["dislikes"]]). \
    transform(df[["dislikes"]])

df["views_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["views"]]). \
    transform(df[["views"]])

def weighted_sorting_score(dataframe, w1=35, w2=35, w3=30):
    return (dataframe["scaled_like"] * w1 / 100 +
            dataframe["scaled_dislike"] * w2 / 100 +
            dataframe["views_scaled"] * w3 / 100)


df["rating"] = weighted_sorting_score(df)

df.sort_values("rating", ascending=False).head(20)


####################
# Sorting by Rating, Comment and Purchase
####################

df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["comment_count"]]). \
    transform(df[["comment_count"]])

def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["comment_count_scaled"] * w1 / 100 +
            dataframe["views_scaled"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)


df["weighted_sorting_score"] = weighted_sorting_score(df)

df.sort_values("weighted_sorting_score", ascending=False).head(20)



df["publish_time"] = pd.to_datetime(df["publish_time"])

df["publish_time"].max()
current_date = pd.to_datetime('2018-06-16 01:31:53+0000')

df["days_diff"] = (current_date - df["publish_time"]).dt.days
df["days_diff"].describe().T
df["rating"].mean()
df.head()

# Time-Based Weighted Average
def time_based_weighted_average(dataframe, w1=25, w2=25, w3=25, w4=25):
    return dataframe.loc[df["days_diff"] <= 30, "rating"].mean() * w1 / 100 + \
        dataframe.loc[(dataframe["days_diff"] > 30) & (dataframe["days_diff"] <= 90), "rating"].mean() * w2 / 100 + \
        dataframe.loc[(dataframe["days_diff"] > 90) & (dataframe["days_diff"] <= 180), "rating"].mean() * w3 / 100 + \
        dataframe.loc[(dataframe["days_diff"] > 180), "rating"].mean() * w4 / 100


time_based_weighted_average(df)
df["rating"].mean()
