import copy
import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# from textblob import TextBlob

# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# this isn't needed


def pretty(s: str) -> str:
    try:
        return dict(js="JavaScript")[s]
    except KeyError:
        return s.capitalize()


## NOT NONSENSE -------------------------------------
st.set_page_config(
    page_title="CART",
    page_icon=None,
    layout="centered",
    # initial_sidebar_state="collapsed",
)


@st.cache_data
def get_data():
    df = pd.read_csv("./data/public_comments_133.csv")
    return df


@st.cache_data
def get_full_data():
    df = pd.read_csv("./data/df_full_wo_emotion.csv")
    # df = pd.read_csv("./data/public_comments_133.csv")
    # clean comments
    # df["clean_comment"] = df["comment"].apply(nlp_clean)
    # df["job"] = df["clean_comment"].apply(get_job)
    return df


@st.cache_data
def get_cosine_sim(df_full):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_full["clean_comment"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


def get_similar_comments(
    index: int, df: pd.DataFrame, cosine_sim, threshold: float
) -> pd.DataFrame:
    """
    Returns a list of comments that are similiar to the comment passed in.
    """
    new_df = df
    new_df["Similar_Score"] = cosine_sim[index].tolist()
    new_df = new_df.loc[new_df["Similar_Score"] > threshold,]

    return new_df


@st.cache_data
def group_similar_comments(df, threshold):
    # start timer
    start = time.time()

    vectorizer = TfidfVectorizer()
    i = 1
    all_df = pd.DataFrame()
    print("Performing TF-IDF")
    df = df.reset_index(drop=True)
    tfidf_matrix = vectorizer.fit_transform(df["clean_comment"])
    print("TF-IDF done")
    print("Performing cosine similarity")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print("Cosine similarity done")
    while len(df) > 0:
        cosine_sim_df = pd.DataFrame(
            cosine_sim[df.index[0]].tolist(), columns=["Similar_Score"]
        )
        df = df.join(cosine_sim_df)

        temp_df = df.loc[df["Similar_Score"] > threshold,]
        temp_df = temp_df[["id", "clean_comment"]]
        temp_df["Unique_Comment_ID"] = i
        print("Similar Comments: ", len(temp_df))
        all_df = pd.concat([all_df, temp_df], ignore_index=True)

        # remove rows from df that are in new_df
        df = df[~df["id"].isin(temp_df["id"])]
        # remove Similar_Score column
        df = df.drop(columns=["Similar_Score"])
        i += 1
        if len(temp_df) == 0:
            print("No more similar comments")
            break

    time_elapsed = time.time() - start
    return all_df, cosine_sim_df, df, time_elapsed


df = get_data()
df_full = copy.deepcopy(get_full_data())
cosine_sim = copy.deepcopy(get_cosine_sim(df_full))

# convert the date to a datetime object
df_full["attributes_posted_date"] = pd.to_datetime(df_full["attributes_posted_date"])

#### SIDE BAR STUFF --------------------------------------------------------------
LOGO = "./img/aiports_logo.png"
st.sidebar.image(LOGO, use_column_width=True)
st.sidebar.text("Jason Lee\nSr. Data Scientist")

number = st.sidebar.number_input(
    "Select Comment",
    step=1,
    min_value=0,
    max_value=len(df_full),
    help="The number corresponds to index number of the original dataset. 0 = First comment, 9 = 10th comment",
    key="comment_number",
)

limit = st.sidebar.number_input(
    "Select Similarity Threshold",
    step=0.01,
    value=0.9,
    min_value=0.0,
    max_value=1.0,
    help="0 to 1 scale with 0 being no similarity and 1.0 being 100% similarity",
    key="threshold",
)


grouped_df, cosine_sim_df, pointless_df, time_elapsed = group_similar_comments(
    df_full, limit
)

st.sidebar.markdown(
    """

**ACSA** is a tool for analyzing comments from the [Public Comments](https://www.regulations.gov/docket/VA-2020-VHA-0024/comments) form.
"""
)


st.title("ACSA")
st.subheader(
    "Automated Comment Similarity Analysis for Streamlining Regulatory Review Processes"
)

st.markdown(
    """
### Summary of Regulation:  
The VA is stating that its healthcare professionals can practice within VA guidelines, regardless of state requirements. This includes providing services in states where they aren't licensed, aiming to improve access to VA healthcare. The rule also emphasizes VA's authority to set national practice standards for uniformity across its medical facilities.


### Comment Statistics:
"""
)

# return how many days difference between the first and last comment
comment_period = (
    df_full["attributes_posted_date"].max() - df_full["attributes_posted_date"].min()
)
# just the number of days
comment_period = f"{comment_period.days} days"
time_elapsed = f"{round(time_elapsed)} sec"


# create several columns to display multiple metrics
a1, a2, a3, a4, a5 = st.columns(5)
a1.metric(label="Comments", value=len(df_full))
a2.metric(label="Unique Comments", value=grouped_df["Unique_Comment_ID"].nunique())
a4.metric(label="Time to Calculate", value=time_elapsed)
a3.metric(label="Comment Period", value=comment_period)
a5.metric(label="Similarity Threshold", value=limit)


df_test = df
import altair as alt

st.markdown("---")


# TEXT SIMILARITY
# clean comments
# df_full["clean_comment"] = df_full["comment"].apply(nlp_clean)
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(df_full["clean_comment"])
# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


sim_comments = get_similar_comments(
    index=number,
    df=df_full,
    cosine_sim=cosine_sim,
    threshold=limit,
)


st.markdown(
    f"""
### Similar Comments


##### Selected Comment:
{df_full["clean_comment"][number]}

##### **Number of similar comments found: {len(sim_comments)}  ~  {round((len(sim_comments) / len(df_full) * 100), 1)}% of the comments**

"""
)


# drop the columns we don't need
sim_comments = sim_comments.drop(
    columns=[
        "type",
        "attributes_document_type",
        "attributes_last_modified_date",
        "attributes_highlighted_content",
        "attributes_withdrawn",
        "attributes_agency_id",
        "attributes_title",
        "attributes_object_id",
        "attributes_posted_date",
        "links_self",
        "comment",
        "sentiment_TEXTBLOB",
        # "sentiment_VADER",
        # "sentiment_FLAIR_label",
        # "sentiment_FLAIR_score",
    ]
)


st.info("Click the box below to view all similar comments")
view_data = st.checkbox("View dataset", value=False)
if view_data:
    st.dataframe(sim_comments)


st.markdown("---")

st.markdown(
    """
### Profession Extraction
"""
)

job_chart = (
    alt.Chart(df_test, title="Professions of Public Comments")
    .mark_bar()
    .encode(
        y=alt.Y(
            "job",
            sort=alt.EncodingSortField(field="job", op="count", order="descending"),
            title="Job",
        ),
        x=alt.X("count(job)", title="Count"),
        # color=alt.Color("job", legend=None, scale=alt.Scale(scheme="Blues")),
        color=alt.Color("job:N", legend=None, scale=alt.Scale(scheme="blues")),
        tooltip=["job", "count(job)"],
    )
    .interactive()
)
st.altair_chart(job_chart, use_container_width=True)

st.markdown("---")

st.markdown(
    """
### Sentiment Analysis
"""
)


sentiment_chart = (
    alt.Chart(df_test, title="Distribution of Sentiment Scores")
    .transform_density(
        "sentiment_TEXTBLOB",
        as_=["sentiment_TEXTBLOB", "density"],
    )
    .mark_area()
    .encode(
        y=alt.Y(
            "density:Q",
            title="",
        ),
        x=alt.X("sentiment_TEXTBLOB:Q", title="Sentiment Score"),
        tooltip=["density:Q"],
    )
    .interactive()
)
st.altair_chart(sentiment_chart, use_container_width=True)

st.markdown("---")

st.markdown(
    """
### Emotion Analysis

**Distribution** of **Emotion** Scores
"""
)


def plot_emotion_dist(df, emotion):
    """ "
    Plots the distribution of the emotion passed in.
    """
    emotion_chart = (
        alt.Chart(df, height=150, title=f"{emotion.upper()}")
        .transform_density(
            emotion,
            as_=[emotion, "density"],
            extent=[0, 1],
        )
        .mark_area()
        .encode(
            x=alt.X(f"{emotion}:Q", title="Score"),
            y=alt.Y(
                "density:Q",
                title="",
            ),
            tooltip=["density:Q"],
        )
        .interactive()
    )

    return emotion_chart


anger_chart = plot_emotion_dist(df, "anger")
disgust_chart = plot_emotion_dist(df, "disgust")
fear_chart = plot_emotion_dist(df, "fear")
joy_chart = plot_emotion_dist(df, "joy")
sadness_chart = plot_emotion_dist(df, "sadness")
surprise_chart = plot_emotion_dist(df, "surprise")
neutral_chart = plot_emotion_dist(df, "neutral")

st.altair_chart(anger_chart, use_container_width=True)
st.altair_chart(disgust_chart, use_container_width=True)
st.altair_chart(fear_chart, use_container_width=True)
st.altair_chart(joy_chart, use_container_width=True)
st.altair_chart(sadness_chart, use_container_width=True)
st.altair_chart(surprise_chart, use_container_width=True)
st.altair_chart(neutral_chart, use_container_width=True)

st.markdown("---")
