"""
Loading and preparing the Twitter US Airline Sentiment data.

14,640 tweets to six US airlines from February 16-24, 2015, each labeled by
crowd workers with a sentiment and, for negative tweets, a complaint reason.
Released by Figure Eight (CrowdFlower) under CC BY-NC-SA 4.0.
"""
import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "Tweets.csv"
SOURCE_URL = "https://raw.githubusercontent.com/albahnsen/AppliedDeepLearningClass/master/datasets/Tweets.zip"

SENTIMENTS = ["negative", "neutral", "positive"]

# Which team a complaint goes to. Crowd workers chose one of ten reasons;
# a customer-care desk has fewer queues than that.
ROUTE_FOR_REASON = {
    "Lost Luggage": "baggage",
    "Damaged Luggage": "baggage",
    "Late Flight": "disruption",
    "Cancelled Flight": "disruption",
    "Flight Booking Problems": "booking",
    "Customer Service Issue": "service",
    "Flight Attendant Complaints": "service",
    "longlines": "service",
    "Bad Flight": "service",
    "Can't Tell": "general",
}
ROUTES = ["no_action", "baggage", "disruption", "booking", "service", "general"]

AIRLINE_HANDLES = {"united", "usairways", "americanair", "southwestair", "jetblue", "virginamerica",
                   "delta", "deltaassist", "virginatlantic", "americanairlines", "southwest"}

# Train on the first six days, tune on the seventh, test on the last two.
# Models always predict on data newer than what they learned from, so the
# test set should be too.
VAL_DAY = "2015-02-22"
TEST_FROM = "2015-02-23"


def load_raw() -> pd.DataFrame:
    if not DATA.exists():
        import io, urllib.request, zipfile
        DATA.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCE_URL) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extract("Tweets.csv", DATA.parent)
    return pd.read_csv(DATA)


def repair_label_leak(text: str) -> str:
    """
    Undo a find-and-replace that was run over this dataset before release:
    "cancel" became "Cancelled Flight" and "late" became "Late Flight", inside
    the tweets themselves. That's why the text reads "Cancelled Flightled" and
    "Late Flightr". It affects about 1,470 tweets, and since those strings are
    the label names, leaving them in lets a model read the answer off the
    question. The replacement was mechanical, so reversing it restores the
    original words: "Cancelled Flightled" -> "cancelled", "Late Flightr" -> "later".
    """
    return text.replace("Cancelled Flight", "cancel").replace("Late Flight", "late")


def normalize(text: str) -> str:
    """
    Hide which airline a tweet is about, so the model learns complaints, not
    brands. Otherwise "@united" alone predicts a lot, because some airlines
    simply got angrier tweets that week.
    """
    t = re.sub(r"https?://\S+", " <url> ", repair_label_leak(text))
    t = re.sub(r"@(\w+)", lambda m: " @airline " if m.group(1).lower() in AIRLINE_HANDLES else " @user ", t)
    t = t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", t).strip()


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["created"] = pd.to_datetime(df["tweet_created"].str.slice(0, 19))
    df = df.sort_values("created")
    # Retweets and copy-pastes: the same text in train and test would inflate every score.
    df["clean"] = df["text"].map(normalize)
    df = df.drop_duplicates("clean", keep="first")
    df["route"] = [
        ROUTE_FOR_REASON.get(r, "general") if s == "negative" else "no_action"
        for s, r in zip(df["airline_sentiment"], df["negativereason"])
    ]
    day = df["created"].dt.strftime("%Y-%m-%d")
    df["split"] = ["test" if d >= TEST_FROM else "val" if d == VAL_DAY else "train" for d in day]
    return df.reset_index(drop=True)
