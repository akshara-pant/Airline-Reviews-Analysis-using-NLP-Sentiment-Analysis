"""
SkyVoice - Airline Passenger Feedback Analyser
Streamlit dashboard for the NLP project (Notebooks 1 to 10).

Run it from the project folder with:
    streamlit run app.py
"""
import html
import json
import re
from pathlib import Path

import altair as alt
import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.stem import WordNetLemmatizer

PROJECT_NAME = "SkyVoice"
TAGLINE = "What do airline passengers really say? Thousands of reviews, turned into clear insights."

st.set_page_config(page_title=PROJECT_NAME + " - Airline Feedback Analyser", page_icon="✈️", layout="wide")

# colours used everywhere
RED, GREY, GREEN, BLUE, ORANGE = "#E4572E", "#B8BEC7", "#2E9E5B", "#1E5AA8", "#F2994A"

TOPIC_NAMES = {"delay": "Delays", "cancellation": "Cancellations", "baggage": "Baggage",
               "boarding_checkin": "Boarding and check-in", "cabin_crew": "Cabin crew",
               "ground_staff": "Ground staff", "seat": "Seats", "food": "Food",
               "entertainment": "Entertainment", "wifi": "WiFi", "value_fees": "Value and fees",
               "refund_compensation": "Refunds and compensation", "communication": "Communication"}
TOPIC_ICONS = {"delay": "⏰", "cancellation": "🚫", "baggage": "🧳", "boarding_checkin": "🛂", "cabin_crew": "🤝",
               "ground_staff": "🏢", "seat": "💺", "food": "🍴", "entertainment": "🎬", "wifi": "📶",
               "value_fees": "💰", "refund_compensation": "💸", "communication": "💬"}
SUGGESTIONS = {
    "delay": "Improve on-time performance, for example with more realistic schedules and enough buffer time on busy routes.",
    "cancellation": "Reduce cancellations, and when they happen rebook passengers quickly with clear options.",
    "baggage": "Improve baggage handling and tracking, and make the process for lost or damaged bags simple and fast.",
    "boarding_checkin": "Make boarding and check-in smoother with better queue management, more counters and self-service options.",
    "cabin_crew": "Strengthen cabin crew training on courtesy and responsiveness.",
    "ground_staff": "Train ground and customer service staff, and shorten the time it takes to answer passengers.",
    "seat": "Review seat comfort and legroom, and be clear about what each seat type offers.",
    "food": "Review the quality, temperature and variety of onboard meals and drinks.",
    "entertainment": "Update the inflight entertainment system and fix broken screens quickly.",
    "wifi": "Improve the reliability and speed of onboard WiFi.",
    "value_fees": "Be transparent about fees and extra charges so that passengers feel they get fair value.",
    "refund_compensation": "Speed up refund and compensation handling, with clear rules and status updates.",
    "communication": "Send proactive messages (SMS or app) about delays, gate changes and cancellations.",
}
SHORT_NAMES = {"delay": "Delays", "cancellation": "Cancellations", "baggage": "Baggage", "boarding_checkin": "Boarding",
               "cabin_crew": "Cabin crew", "ground_staff": "Ground staff", "seat": "Seats", "food": "Food",
               "entertainment": "Entertainment", "wifi": "WiFi", "value_fees": "Value / fees", "refund_compensation": "Refunds",
               "communication": "Communication"}
EXAMPLE_REVIEWS = {
    "😀 Happy passenger": "Excellent flight from start to finish. The cabin crew were friendly and attentive, the food was delicious "
                          "and the seats were very comfortable. Boarding was smooth and we landed on time. Would definitely fly again.",
    "😠 Unhappy passenger": "Terrible experience. Our flight was delayed for five hours and nobody told us anything. Then they lost my "
                            "luggage and the customer service refused to give a refund. I will never fly with them again.",
    "😐 Mixed review": "The crew were lovely and the food was good, but the flight was delayed for three hours and we got no "
                       "information at the gate. The seat was cramped and the entertainment screen did not work.",
}
PAGES = ["🏠 Overview", "🚨 Top Problems", "⚖️ Compare Airlines", "📋 Airline Report Card", "📝 Review Analyser", "ℹ️ About SkyVoice"]

REQUIRED_FILES = ["data/reviews_preprocessed.csv", "data/aspect_reviews.csv", "data/aspect_segments.csv",
                  "data/topic_priority.csv", "data/airline_topic_rates.csv", "data/recommendations.csv",
                  "data/airline_summary.csv", "data/lexrank_summaries.csv", "data/topic_words.json",
                  "models/sentiment_model.joblib"]

STYLE = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1250px;}
.hero {background: linear-gradient(100deg, #0B3D91 0%, #1E88E5 100%); color: white; padding: 18px 26px;
       border-radius: 14px; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 2.1rem; color: white; padding: 0;}
.hero p {margin: 4px 0 0 0; color: #E3F2FD; font-size: 1.02rem;}
.hero .facts {margin-top: 8px; font-size: 0.85rem; color: #BBDEFB;}
.card {background: white; border: 1px solid #E5E9F0; border-radius: 12px; padding: 12px 16px; height: 100%;
       box-shadow: 0 1px 3px rgba(0,0,0,0.06);}
.card-title {font-size: 0.75rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.05em;}
.card-value {font-size: 1.6rem; font-weight: 700; color: #1F2933; line-height: 1.25;}
.card-note {font-size: 0.85rem; color: #6B7280;}
.tile {background: white; border: 1px solid #E5E9F0; border-radius: 10px; padding: 8px 12px; margin-bottom: 10px;}
.tile-name {font-size: 0.9rem; color: #1F2933; font-weight: 600;}
.tile-value {font-size: 1.4rem; font-weight: 700; color: #1F2933;}
.tile-note {font-size: 0.78rem; color: #6B7280;}
.chip {display: inline-block; padding: 3px 11px; margin: 3px 5px 3px 0; border-radius: 999px; font-size: 0.85rem;}
.review-box {line-height: 2.1; font-size: 1.02rem; padding: 14px 18px; background: white; border: 1px solid #E5E9F0;
             border-radius: 12px;}
.review-box span.neg {background: #FBD9D0; border-radius: 5px; padding: 2px 4px;}
.review-box span.pos {background: #D5F0DE; border-radius: 5px; padding: 2px 4px;}
.review-box span.neu {background: #ECEEF1; border-radius: 5px; padding: 2px 4px;}
.big-verdict {font-size: 1.3rem; font-weight: 700; margin: 6px 0 12px 0;}
</style>
"""


# ================================================================ loading data
@st.cache_data
def load_reviews():
    # the row number of a review is its review_id in all the other files
    return pd.read_csv("data/reviews_preprocessed.csv", usecols=["airline", "year", "recommended_bin", "seat_type"])


@st.cache_data
def load_full_reviews():
    return pd.read_csv("data/reviews_preprocessed.csv",
                       usecols=["airline", "year", "recommended", "over_all_rating", "review_text"])


@st.cache_data
def load_aspects():
    return pd.read_csv("data/aspect_reviews.csv", usecols=["review_id", "topic", "label"])


@st.cache_data
def load_pieces():
    return pd.read_csv("data/aspect_segments.csv", usecols=["review_id", "topic", "text", "learned"])


@st.cache_data
def load_table(name, index_col=None):
    return pd.read_csv("data/" + name + ".csv", index_col=index_col)


@st.cache_resource
def load_model():
    nltk.download("wordnet", quiet=True)
    saved = joblib.load("models/sentiment_model.joblib")
    return {"tfidf": saved["tfidf"], "model": saved["model"],
            "topic_words": json.load(open("data/topic_words.json")),
            "lemmatizer": WordNetLemmatizer(), "lemma_cache": {},
            "features": saved["tfidf"].get_feature_names_out()}


# ================================================================ small helpers
def name_of(topic):
    return TOPIC_NAMES.get(topic, topic)


def pretty(topic):
    if topic not in TOPIC_NAMES:          # already a nice name (or unknown): leave it as it is
        return topic
    return TOPIC_ICONS.get(topic, "") + " " + TOPIC_NAMES[topic]


def card(title, value, note="", color=BLUE):
    st.markdown('<div class="card" style="border-top: 4px solid ' + color + ';"><div class="card-title">' + title +
                '</div><div class="card-value">' + str(value) + '</div><div class="card-note">' + note + '</div></div>',
                unsafe_allow_html=True)


def banner(subtitle):
    reviews = load_reviews()
    facts = (f"{len(reviews):,} reviews · {reviews['airline'].nunique()} airlines · "
             f"{int(reviews['year'].min())} to {int(reviews['year'].max())}")
    st.markdown('<div class="hero"><h1>✈️ ' + PROJECT_NAME + '</h1><p>' + subtitle + '</p><div class="facts">' + facts +
                '</div></div>', unsafe_allow_html=True)


def page_intro(title, explanation):
    st.header(title)
    st.markdown(explanation)


def topic_shares(aspects, review_ids):
    # for every topic: how many comments are unhappy / neutral / happy
    data = aspects[aspects["review_id"].isin(review_ids)]
    shares = data.groupby("topic")["label"].value_counts().unstack(fill_value=0)
    for label in ["negative", "neutral", "positive"]:
        if label not in shares.columns:
            shares[label] = 0
    shares["mentions"] = shares[["negative", "neutral", "positive"]].sum(axis=1)
    for label in ["negative", "neutral", "positive"]:
        shares["% " + label] = shares[label] / shares["mentions"] * 100
    return shares


def sentiment_chart(shares):
    # stacked bars: unhappy / neutral / happy for each topic
    rows = []
    for topic, row in shares.iterrows():
        for label, word, order in [("negative", "Unhappy", 1), ("neutral", "Neutral", 2), ("positive", "Happy", 3)]:
            rows.append({"Topic": pretty(topic), "Feeling": word, "order": order,
                         "Percent": round(row["% " + label], 1), "Comments": int(row[label])})
    long = pd.DataFrame(rows)
    topic_order = [pretty(t) for t in shares.sort_values("% negative", ascending=False).index]
    return (alt.Chart(long).mark_bar()
            .encode(y=alt.Y("Topic:N", sort=topic_order, title=None),
                    x=alt.X("Percent:Q", title="% of comments about the topic", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("Feeling:N", scale=alt.Scale(domain=["Unhappy", "Neutral", "Happy"], range=[RED, GREY, GREEN]),
                                    legend=alt.Legend(orient="bottom", title=None)),
                    order=alt.Order("order:Q"),
                    tooltip=["Topic", "Feeling", alt.Tooltip("Percent:Q", format=".0f"), "Comments"])
            .properties(height=max(120, 30 * len(topic_order))))


def trend_chart(selected_by_year, overall_by_year=None):
    one = selected_by_year.rename("Rate").reset_index()
    one["Group"] = "Selected reviews"
    data = one
    if overall_by_year is not None:
        other = overall_by_year.rename("Rate").reset_index()
        other["Group"] = "All airlines"
        data = pd.concat([one, other])
    data["Rate"] = data["Rate"].round(1)
    return (alt.Chart(data).mark_line(point=True, strokeWidth=3)
            .encode(x=alt.X("year:O", title="Year", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Rate:Q", title="% who would recommend", scale=alt.Scale(domain=[0, max(60, float(data["Rate"].max()) + 5)])),
                    color=alt.Color("Group:N", scale=alt.Scale(domain=["Selected reviews", "All airlines"], range=[BLUE, GREY]),
                                    legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["year", "Group", alt.Tooltip("Rate:Q", format=".1f", title="% recommend")])
            .properties(height=300))


def ranking_chart(plot):
    # bars: which problems have the highest priority score
    return (alt.Chart(plot).mark_bar(color=RED)
            .encode(y=alt.Y("Problem:N", sort="-x", title=None), x=alt.X("Priority score:Q", title="Priority score"),
                    tooltip=["Problem", "Priority score", alt.Tooltip("Common (% of reviews):Q", format=".1f"),
                             alt.Tooltip("Damage (points):Q", format=".1f")])
            .properties(height=380))


def common_vs_damaging_chart(plot):
    # dots: how common (x) against how damaging (y)
    dots = alt.Chart(plot).mark_circle(size=170, color=BLUE, opacity=0.85).encode(
        x=alt.X("Common (% of reviews):Q", title="How common (% of reviews with the complaint)"),
        y=alt.Y("Damage (points):Q", title="How damaging (drop in recommend rate)", scale=alt.Scale(zero=False)),
        tooltip=["Problem", alt.Tooltip("Common (% of reviews):Q", format=".1f"), alt.Tooltip("Damage (points):Q", format=".1f")])
    # label positions are chosen by hand, so that crowded dots do not hide each other's names
    positions = {"ground_staff": "above", "value_fees": "below", "delay": "above", "boarding_checkin": "right",
                 "baggage": "below", "communication": "above", "refund_compensation": "below", "cancellation": "above",
                 "cabin_crew": "above", "food": "below", "seat": "below", "entertainment": "below", "wifi": "above"}
    plot = plot.copy()
    plot["position"] = plot["topic"].map(positions).fillna("above")
    plot["Short name"] = plot["topic"].map(SHORT_NAMES)
    layers = [dots]
    for position, settings in [("above", {"dy": -11}), ("below", {"dy": 14}), ("right", {"dx": 9, "align": "left"})]:
        layers.append(alt.Chart(plot).mark_text(fontSize=11, **settings).transform_filter(alt.datum.position == position).encode(
            x="Common (% of reviews):Q", y=alt.Y("Damage (points):Q", scale=alt.Scale(zero=False)), text="Short name:N"))
    return alt.layer(*layers).properties(height=380)


def compare_bars(long, order):
    return (alt.Chart(long).mark_bar()
            .encode(y=alt.Y("Topic:N", sort=order, title=None), x=alt.X("Complaint rate:Q", title="% of reviews with a complaint"),
                    yOffset="Airline:N", color=alt.Color("Airline:N", scale=alt.Scale(range=["#1E5AA8", "#8E44AD", "#16A085", "#7F8C8D"]),
                                                         legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["Airline", "Topic", "Complaint rate"])
            .properties(height=max(300, 55 * len(order))))


def compare_heatmap(long, order):
    base = alt.Chart(long).encode(x=alt.X("Airline:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("Topic:N", sort=order, title=None))
    heat = base.mark_rect().encode(color=alt.Color("Complaint rate:Q", scale=alt.Scale(scheme="reds"), legend=None),
                                   tooltip=["Airline", "Topic", "Complaint rate"])
    numbers = base.mark_text(fontSize=12).encode(text=alt.Text("Complaint rate:Q", format=".0f"),
                                                 color=alt.condition(alt.datum["Complaint rate"] > 20, alt.value("white"), alt.value("black")))
    return (heat + numbers).properties(height=max(300, 32 * len(order)))


def show_examples(pieces, topic, kind, count=5, seed=0):
    # prints a few real comments (kind = "unhappy" or "happy")
    subset = pieces[pieces["topic"] == topic]
    subset = subset[subset["learned"] < 0.35] if kind == "unhappy" else subset[subset["learned"] >= 0.65]
    subset = subset[subset["text"].str.split().str.len().between(8, 40)]
    if len(subset) == 0:
        st.caption("No comments found for this selection.")
        return
    for text in subset.sample(min(count, len(subset)), random_state=seed)["text"]:
        st.markdown("> *" + text.replace("\n", " ") + "*")


def next_examples():
    st.session_state["seed"] = st.session_state.get("seed", 0) + 1


# ================================================================ page 1: overview
def show_overview():
    page_intro("🏠 Overview",
               "**What you are seeing:** how passengers feel about different parts of the flight. "
               "Pick an airline, a class or a time period to zoom in.")

    reviews = load_reviews()
    aspects = load_aspects()
    pieces = load_pieces()

    column1, column2, column3 = st.columns([2, 2, 3])
    airline = column1.selectbox("✈️ Airline", ["All airlines"] + sorted(reviews["airline"].unique()), key="ov_airline")
    seat_class = column2.selectbox("💺 Travel class", ["All classes"] + sorted(reviews["seat_type"].dropna().unique()), key="ov_class")
    first_year, last_year = int(reviews["year"].min()), int(reviews["year"].max())
    years = column3.slider("📅 Years", first_year, last_year, (first_year, last_year), key="ov_years")

    base = reviews[reviews["year"].between(years[0], years[1])]
    if seat_class != "All classes":
        base = base[base["seat_type"] == seat_class]
    selected = base if airline == "All airlines" else base[base["airline"] == airline]

    if len(selected) < 5:
        st.warning("There are almost no reviews for this selection. Try a wider choice.")
        return

    shares = topic_shares(aspects, selected.index)
    min_mentions = 100 if len(selected) >= 2000 else 15
    reliable = shares[shares["mentions"] >= min_mentions]
    rate = selected["recommended_bin"].mean() * 100
    overall_rate = base["recommended_bin"].mean() * 100

    # ---- key numbers
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        card("Reviews analysed", f"{len(selected):,}", f"{selected['airline'].nunique()} airline(s)")
    with k2:
        note = "" if airline == "All airlines" else f"all airlines: {overall_rate:.0f}%"
        card("Would recommend", f"{rate:.0f}%", note, GREEN if rate >= 50 else (ORANGE if rate >= 30 else RED))
    if len(reliable) > 0:
        with k3:
            complaint = reliable["negative"].idxmax()
            card("Most common complaint", pretty(complaint), f"{int(reliable.loc[complaint, 'negative']):,} unhappy comments", RED)
        with k4:
            praise = reliable["positive"].idxmax()
            card("Most praised", pretty(praise), f"{int(reliable.loc[praise, 'positive']):,} happy comments", GREEN)

    # ---- plain-language summary
    st.write("")
    with st.container(border=True):
        st.markdown("**📌 In short**")
        st.markdown(f"- **{rate:.0f}%** of these passengers would recommend the airline" +
                    ("" if airline == "All airlines" else f" (all airlines: {overall_rate:.0f}%)") + ".")
        if len(reliable) >= 4:
            worst = reliable.sort_values("% negative", ascending=False).head(3).index
            best = reliable.sort_values("% positive", ascending=False).head(2).index
            st.markdown("- Passengers are most **unhappy** about: " + ", ".join(f"**{name_of(t)}** ({reliable.loc[t, '% negative']:.0f}% negative)" for t in worst) + ".")
            st.markdown("- They are happiest about: " + " and ".join(f"**{name_of(t)}** ({reliable.loc[t, '% positive']:.0f}% positive)" for t in best) + ".")
        by_year = selected.groupby("year")["recommended_bin"].mean() * 100
        if len(by_year) >= 2:
            direction = "went up" if by_year.iloc[-1] > by_year.iloc[0] else "went down"
            st.markdown(f"- The share who recommend **{direction}** from {by_year.iloc[0]:.0f}% in {by_year.index[0]} to {by_year.iloc[-1]:.0f}% in {by_year.index[-1]}.")

    # ---- charts
    left, right = st.columns(2)
    with left:
        st.subheader("Are passengers getting happier?")
        st.caption("Share of passengers who would recommend the airline, by year. Hover over a point for details.")
        overall_by_year = None if airline == "All airlines" else base.groupby("year")["recommended_bin"].mean() * 100
        st.altair_chart(trend_chart(by_year, overall_by_year))
    with right:
        st.subheader("How do they feel about each part of the flight?")
        st.caption("Red = unhappy, grey = neutral, green = happy. Sorted from most to least unhappy.")
        chart_data = shares[shares["mentions"] >= min_mentions]
        if len(chart_data) == 0:
            st.info("Too few comments in this selection to show the topics.")
        else:
            st.altair_chart(sentiment_chart(chart_data))

    # ---- read real comments
    st.subheader("🔍 Read what passengers actually say")
    topics_here = list(reliable.sort_values("negative", ascending=False).index) if len(reliable) > 0 else list(shares.index)
    if len(topics_here) > 0:
        column1, column2 = st.columns([3, 1])
        topic = column1.selectbox("Choose a topic", topics_here, format_func=pretty, key="ov_topic")
        column2.write("")
        column2.button("🔄 Show other comments", on_click=next_examples, key="ov_next")
        seed = st.session_state.get("seed", 0)
        chosen_pieces = pieces[pieces["review_id"].isin(selected.index)]
        left, right = st.columns(2)
        with left:
            st.markdown("##### 😠 Unhappy comments")
            show_examples(chosen_pieces, topic, "unhappy", seed=seed)
        with right:
            st.markdown("##### 😀 Happy comments")
            show_examples(chosen_pieces, topic, "happy", seed=seed)


# ================================================================ page 2: top problems
def show_problems():
    page_intro("🚨 Top Problems",
               "**What you are seeing:** which problems matter most to an airline. A problem matters when it is "
               "**common** (many passengers have it) *and* **damaging** (those passengers stop recommending the airline).")

    priority = load_table("topic_priority", index_col=0).sort_values("priority score", ascending=False)
    pieces = load_pieces()
    reviews = load_reviews()

    # top 3 as big cards
    columns = st.columns(3)
    for rank, (topic, row) in enumerate(priority.head(3).iterrows(), start=1):
        with columns[rank - 1]:
            card("Problem #" + str(rank), pretty(topic),
                 f"{row['% of all reviews']:.0f}% of passengers complain · only {row['recommend rate with complaint']:.0f}% of them still recommend", RED)

    plot = priority.reset_index().rename(columns={"index": "topic"})
    plot["Problem"] = plot["topic"].map(pretty)
    plot["Priority score"] = plot["priority score"]
    plot["Common (% of reviews)"] = plot["% of all reviews"]
    plot["Damage (points)"] = plot["drop (points)"]

    left, right = st.columns(2)
    with left:
        st.subheader("Ranking")
        st.caption("Priority score = how common x how damaging. Longer bar = more important to fix.")
        st.altair_chart(ranking_chart(plot))
    with right:
        st.subheader("Common or damaging?")
        st.caption("Top right = common AND damaging. Hover over a dot for details.")
        st.altair_chart(common_vs_damaging_chart(plot))

    with st.expander("📊 See all the numbers"):
        shown = priority[["% of all reviews", "recommend rate with complaint", "recommend rate without", "drop (points)", "priority score"]].copy()
        shown.index = [name_of(t) for t in shown.index]
        st.dataframe(shown)
    st.caption("Note: ground staff and value / fees are the least reliable topics (see Notebook 7), so read them with some care.")

    # deep dive
    st.subheader("🔍 Look closer at one problem")
    column1, column2 = st.columns(2)
    topic = column1.selectbox("Problem", list(priority.index), format_func=pretty, key="pr_topic")
    airline = column2.selectbox("Airline", ["All airlines"] + sorted(reviews["airline"].unique()), key="pr_airline")

    row = priority.loc[topic]
    c1, c2, c3 = st.columns(3)
    with c1:
        card("Passengers with this complaint", f"{int(row['reviews with complaint']):,}", f"{row['% of all reviews']:.0f}% of all reviews", RED)
    with c2:
        card("Still recommend (with complaint)", f"{row['recommend rate with complaint']:.1f}%", "", RED)
    with c3:
        card("Still recommend (without)", f"{row['recommend rate without']:.1f}%", "", GREEN)

    if airline != "All airlines":
        pieces = pieces[pieces["review_id"].isin(reviews[reviews["airline"] == airline].index)]
    st.write("")
    column1, column2 = st.columns([4, 1])
    column1.markdown("**Real unhappy comments about " + name_of(topic).lower() + "**")
    column2.button("🔄 Other comments", on_click=next_examples, key="pr_next")
    show_examples(pieces, topic, "unhappy", seed=st.session_state.get("seed", 0))


# ================================================================ page 3: compare airlines
def show_compare():
    page_intro("⚖️ Compare Airlines",
               "**What you are seeing:** how often each airline's passengers complain about each topic. "
               "A lower bar is better. Choose 2 to 4 airlines (only airlines with at least 300 reviews are included).")

    rates = load_table("airline_topic_rates", index_col=0)
    topics = [c for c in rates.columns if c not in ("reviews", "recommend rate (%)")]
    average = rates[topics].mean()
    names = sorted(rates.index)

    default = [a for a in ["Emirates", "American Airlines"] if a in names]
    chosen = st.multiselect("Choose airlines to compare", names, default=default, max_selections=4, key="cmp_airlines")
    if len(chosen) < 2:
        st.info("Please choose at least two airlines.")
        return

    columns = st.columns(len(chosen))
    average_recommend = rates["recommend rate (%)"].mean()
    for column, airline in zip(columns, chosen):
        value = rates.loc[airline, "recommend rate (%)"]
        with column:
            card(airline, f"{value:.0f}% recommend", f"{int(rates.loc[airline, 'reviews']):,} reviews · average airline: {average_recommend:.0f}%",
                 GREEN if value >= average_recommend else RED)

    long = rates.loc[chosen, topics].reset_index().melt(id_vars="airline", var_name="topic", value_name="Complaint rate")
    long["Topic"] = long["topic"].map(pretty)
    long["Airline"] = long["airline"]
    long["Complaint rate"] = long["Complaint rate"].round(1)
    order = [pretty(t) for t in rates.loc[chosen, topics].mean().sort_values(ascending=False).index]

    tab1, tab2 = st.tabs(["📊 Bar chart", "🔥 Heatmap"])
    with tab1:
        st.caption("% of the airline's reviews that complain about the topic. Hover for details.")
        st.altair_chart(compare_bars(long, order))
    with tab2:
        st.caption("Darker red = more complaints. The number is the % of reviews with a complaint.")
        st.altair_chart(compare_heatmap(long, order))

    st.subheader("In plain words")
    for airline in chosen:
        ratio = (rates.loc[airline, topics] / average).sort_values()
        best, worst = ratio.index[:2], ratio.index[-2:][::-1]
        st.markdown(f"- **{airline}** gets fewer complaints than the average airline about " +
                    " and ".join(f"**{name_of(t).lower()}**" for t in best) +
                    ", and more complaints about " + " and ".join(f"**{name_of(t).lower()}**" for t in worst) + ".")


# ================================================================ page 4: airline report card
def show_report_card():
    page_intro("📋 Airline Report Card",
               "**What you are seeing:** a one-page summary for one airline: where it is worse or better than the average "
               "airline, what to fix first, and what passengers say.")

    recommendations = load_table("recommendations")
    summary = load_table("airline_summary", index_col=0)
    lexrank = load_table("lexrank_summaries")
    rates = load_table("airline_topic_rates", index_col=0)
    topics = [c for c in rates.columns if c not in ("reviews", "recommend rate (%)")]
    average = rates[topics].mean()

    names = sorted(summary.index)
    airline = st.selectbox("✈️ Choose an airline", names, index=names.index("American Airlines") if "American Airlines" in names else 0, key="rc_airline")
    info = summary.loc[airline]
    average_recommend = rates["recommend rate (%)"].mean()

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Reviews", f"{int(info['reviews']):,}", "")
    with c2:
        value = info["recommend rate (%)"]
        card("Would recommend", f"{value:.0f}%", f"average airline: {average_recommend:.0f}%", GREEN if value >= average_recommend else RED)
    with c3:
        worse = int(sum(rates.loc[airline, t] > average[t] for t in topics))
        card("Topics worse than average", f"{worse} of {len(topics)}", "", RED if worse > 8 else (ORANGE if worse > 4 else GREEN))

    if isinstance(info["special_note"], str) and info["special_note"] != "":
        st.warning("**📣 Communication during disruptions:** " + info["special_note"])

    # ---- report card grid
    st.subheader("Report card")
    st.caption("% of this airline's reviews that complain about each topic, compared with the average airline. "
               "🔴 much worse (25% or more) · 🟠 worse · 🟢 better or equal.")
    ratio = (rates.loc[airline, topics] / average).sort_values(ascending=False)
    for start in range(0, len(ratio), 4):
        columns = st.columns(4)
        for column, topic in zip(columns, ratio.index[start:start + 4]):
            r = ratio[topic]
            color, mark = (RED, "🔴") if r >= 1.25 else ((ORANGE, "🟠") if r > 1.0 else (GREEN, "🟢"))
            column.markdown('<div class="tile" style="border-left: 6px solid ' + color + ';"><div class="tile-name">' + pretty(topic) +
                            '</div><div class="tile-value">' + f"{rates.loc[airline, topic]:.0f}%" + '</div><div class="tile-note">' + mark +
                            f" average airline: {average[topic]:.0f}%</div></div>", unsafe_allow_html=True)

    # ---- priorities
    rows = recommendations[recommendations["airline"] == airline].sort_values("rank")
    st.subheader("🎯 What to fix first")
    if len(rows) == 0:
        st.info("No topic has at least 30 complaints for this airline, so there is not enough evidence for a recommendation. That is a good sign.")
    for _, r in rows.iterrows():
        with st.expander("Priority " + str(int(r["rank"])) + ": " + r["topic_name"] + "   [" + r["level"] + "]", expanded=(r["rank"] == 1)):
            st.write(f"{r['complaint_rate']:.0f}% of reviews complain about {r['topic_name'].lower()} (average airline: {r['average_rate']:.0f}%).")
            if isinstance(r["key_phrases"], str) and r["key_phrases"] != "":
                chips = "".join('<span class="chip" style="background:#FBD9D0;">' + html.escape(p.strip()) + "</span>" for p in r["key_phrases"].split(","))
                st.markdown("**Frequent complaint phrases:**<br>" + chips, unsafe_allow_html=True)
            st.success("**💡 Suggested action:** " + r["suggestion"])
            said = lexrank[(lexrank["airline"] == airline) & (lexrank["topic"] == r["topic"]) & (lexrank["kind"] == "negative")]
            if len(said) > 0:
                st.markdown("**What unhappy passengers say** (summary of " + str(int(said["sentences"].iloc[0])) + " comments):")
                for sentence in said.sort_values("rank")["summary_sentence"]:
                    st.markdown("> *" + sentence + "*")

    # ---- strengths
    st.subheader("⭐ Strengths")
    strengths = [t for t in ratio.sort_values().index if ratio[t] <= 0.85][:3]
    shown_any = False
    for topic in strengths:
        said = lexrank[(lexrank["airline"] == airline) & (lexrank["topic"] == topic) & (lexrank["kind"] == "positive")]
        if len(said) > 0:
            shown_any = True
            with st.expander(pretty(topic) + f" - only {rates.loc[airline, topic]:.0f}% complain (average airline: {average[topic]:.0f}%)"):
                for sentence in said.sort_values("rank")["summary_sentence"]:
                    st.markdown("> *" + sentence + "*")
    if not shown_any:
        st.caption("No clear strengths found: no topic is clearly better than the average airline.")

    # ---- download
    lines = ["SKYVOICE REPORT: " + airline.upper(), f"{int(info['reviews'])} reviews, {info['recommend rate (%)']:.0f}% would recommend", ""]
    for _, r in rows.iterrows():
        lines += [f"Priority {int(r['rank'])}: {r['topic_name']} [{r['level']}]",
                  f"  {r['complaint_rate']:.0f}% of reviews complain (average airline: {r['average_rate']:.0f}%)",
                  "  Frequent phrases: " + str(r["key_phrases"]), "  Suggested action: " + r["suggestion"], ""]
    st.download_button("⬇️ Download this report (.txt)", "\n".join(lines), file_name=airline.replace(" ", "_") + "_report.txt", key="rc_download")


# ================================================================ page 5: review analyser
def to_lemmas(text, bundle):
    text = text.lower().replace("’", "'")
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"\b(did|does|do|was|were|is|are|would|could|should|have|has|had|can)nt\b", r"\1 not", text)
    text = text.replace("on board", "onboard")
    text = re.sub(r"[^a-z\s]", " ", text)
    lemmas = []
    for word in text.split():
        if word not in bundle["lemma_cache"]:
            if word == "ground":
                bundle["lemma_cache"][word] = word
            else:
                lemma = bundle["lemmatizer"].lemmatize(word, pos="v")
                bundle["lemma_cache"][word] = bundle["lemmatizer"].lemmatize(lemma, pos="n")
        lemmas.append(bundle["lemma_cache"][word])
    return lemmas


def find_topics(text, bundle):
    lemmas = to_lemmas(text, bundle)
    lemma_set = set(lemmas)
    joined = " " + " ".join(lemmas) + " "
    found = []
    for topic, entry in bundle["topic_words"].items():
        if any(w in lemma_set for w in entry["words"]) or any(" " + p + " " in joined for p in entry["phrases"]):
            found.append(topic)
    return found


def split_review(text, bundle):
    # sentences -> clauses -> (piece of text, topic)
    sentence_split = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“‘(])|\n+')
    clause_split = re.compile(r";\s+|\s+[-–—]\s+|,?\s+(?:but|however|although|though|whereas)\s+", re.I)
    results = []
    for sentence in sentence_split.split(text):
        for clause in clause_split.split(sentence):
            clause = clause.strip(" ,.-")
            if len(clause.split()) < 3:
                continue
            topics = find_topics(clause, bundle)
            if len(topics) >= 2:
                for part in re.split(r",\s+(?:and\s+)?", clause):
                    part = part.strip(" ,.-")
                    if len(part.split()) >= 2:
                        for topic in find_topics(part, bundle):
                            results.append((part, topic))
            else:
                for topic in topics:
                    results.append((clause, topic))
    return results


def label_of(score, neg_cut, pos_cut):
    return "negative" if score < neg_cut else ("positive" if score >= pos_cut else "neutral")


def analyse(text, bundle, neg_cut, pos_cut):
    tfidf, model = bundle["tfidf"], bundle["model"]
    overall = model.predict_proba(tfidf.transform([text]))[0][1]

    rows = []
    for piece, topic in split_review(text, bundle):
        score = model.predict_proba(tfidf.transform([piece]))[0][1]
        rows.append({"topic": topic, "text": piece, "score": score, "label": label_of(score, neg_cut, pos_cut)})
    pieces = pd.DataFrame(rows, columns=["topic", "text", "score", "label"])

    # which words pushed the overall prediction up or down?
    vector = tfidf.transform([text])
    contribution = vector.data * model.coef_[0][vector.indices]
    words = pd.DataFrame({"word": bundle["features"][vector.indices], "effect": contribution}).sort_values("effect")
    return overall, pieces, words


def highlighted_review(text, pieces):
    # colour every piece of the review that belongs to a topic
    spans, position = [], 0
    for piece_text in pieces["text"].drop_duplicates():
        part = pieces[pieces["text"] == piece_text]
        start = text.find(piece_text, position)
        if start == -1:
            start = text.find(piece_text)
        if start == -1:
            continue
        spans.append((start, start + len(piece_text), part["label"].iloc[0], ", ".join(name_of(t) for t in part["topic"])))
        position = start + len(piece_text)
    spans.sort()

    output, last = "", 0
    for start, end, label, topics in spans:
        if start < last:
            continue
        css = {"negative": "neg", "positive": "pos", "neutral": "neu"}[label]
        output += html.escape(text[last:start]) + '<span class="' + css + '" title="' + html.escape(topics) + '">' + html.escape(text[start:end]) + "</span>"
        last = end
    return '<div class="review-box">' + (output + html.escape(text[last:])).replace("\n", "<br>") + "</div>"


def gauge_html(probability):
    percent = probability * 100
    return ('<div style="position:relative; height:20px; border-radius:10px; margin-top:6px; '
            'background:linear-gradient(90deg,#E4572E,#F2C94C,#2E9E5B);">'
            '<div style="position:absolute; left:calc(' + f"{percent:.1f}" + '% - 3px); top:-5px; width:6px; height:30px; '
            'background:#1F2933; border-radius:3px;"></div></div>'
            '<div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#6B7280; margin-top:4px;">'
            '<span>would not recommend</span><span>would recommend</span></div>')


def set_review_text(text):
    st.session_state["review_text"] = text
    st.session_state["real_review"] = None


def pick_random_review():
    full = load_full_reviews()
    candidates = full[full["review_text"].str.len().between(250, 900)]
    row = candidates.sample(1).iloc[0]
    st.session_state["review_text"] = row["review_text"]
    st.session_state["real_review"] = {"text": row["review_text"], "airline": row["airline"], "year": int(row["year"]),
                                       "recommended": row["recommended"], "rating": row["over_all_rating"]}


def show_analyser():
    page_intro("📝 Review Analyser",
               "**Try the system yourself.** Write or paste any airline review. The system tells you if the passenger would "
               "recommend the airline, and how they feel about each topic. Not sure what to write? Use one of the buttons.")

    bundle = load_model()
    st.session_state.setdefault("review_text", EXAMPLE_REVIEWS["😐 Mixed review"])
    st.session_state.setdefault("real_review", None)

    columns = st.columns(4)
    for i, (label, text) in enumerate(EXAMPLE_REVIEWS.items()):
        columns[i].button(label, on_click=set_review_text, args=(text,), key="example_" + str(i), width="stretch")
    columns[3].button("🎲 Random real review", on_click=pick_random_review, key="random_review", width="stretch")

    text = st.text_area("Review text", key="review_text", height=150)

    with st.expander("⚙️ Settings: how strict should the system be?"):
        st.caption("A comment counts as **unhappy** below the first value and as **happy** from the second value. "
                   "Everything in between is neutral.")
        neg_cut, pos_cut = st.slider("Neutral zone", 0.10, 0.90, (0.35, 0.65), key="cuts")

    if len(text.split()) < 3:
        st.info("Please write a review of at least a few words.")
        return

    overall, pieces, words = analyse(text, bundle, neg_cut, pos_cut)

    # ---- overall result and highlighted text
    left, right = st.columns([2, 3])
    with left:
        st.subheader("Overall")
        if overall < neg_cut:
            verdict, color = "🔴 Would NOT recommend", RED
        elif overall >= pos_cut:
            verdict, color = "🟢 Would recommend", GREEN
        else:
            verdict, color = "🟡 Mixed / undecided", ORANGE
        st.markdown('<div class="big-verdict" style="color:' + color + ';">' + verdict + "</div>", unsafe_allow_html=True)
        st.markdown(f"Chance of recommending: **{overall * 100:.0f}%**")
        st.markdown(gauge_html(overall), unsafe_allow_html=True)
    with right:
        st.subheader("Your review, colour-coded")
        if len(pieces) > 0:
            st.markdown(highlighted_review(text, pieces), unsafe_allow_html=True)
            st.markdown('<span class="chip" style="background:#FBD9D0;">unhappy</span><span class="chip" style="background:#ECEEF1;">neutral</span>'
                        '<span class="chip" style="background:#D5F0DE;">happy</span> <span style="font-size:0.8rem;color:#6B7280;">'
                        "Hover over a coloured part to see its topic.</span>", unsafe_allow_html=True)
        else:
            st.info("No specific topic (seat, food, delay...) was found in this text. The overall result on the left is still valid.")

    real = st.session_state.get("real_review")
    if real and real["text"] == text:
        said = "recommended" if real["recommended"] == "yes" else "did NOT recommend"
        st.info(f"🎲 This is a real review of **{real['airline']}** ({real['year']}). The passenger gave a rating of "
                f"{real['rating']:.0f}/10 and **{said}** the airline.")

    # ---- topic by topic
    if len(pieces) > 0:
        st.subheader("Topic by topic")
        topic_scores = pieces.groupby("topic").agg(score=("score", "mean"), text=("text", "first")).sort_values("score")
        columns = st.columns(3)
        for i, (topic, row) in enumerate(topic_scores.iterrows()):
            label = label_of(row["score"], neg_cut, pos_cut)
            face, color = {"negative": ("😠 Unhappy", RED), "neutral": ("😐 Neutral", GREY), "positive": ("😀 Happy", GREEN)}[label]
            with columns[i % 3]:
                with st.container(border=True):
                    st.markdown("**" + pretty(topic) + "**")
                    st.markdown('<span style="color:' + color + '; font-weight:600;">' + face + "</span>", unsafe_allow_html=True)
                    st.progress(float(np.clip(row["score"], 0, 1)))
                    st.caption('"' + row["text"] + '"')

    # ---- why, and what to do
    left, right = st.columns(2)
    with left:
        st.subheader("🧠 Why this result?")
        st.caption("The words that pushed the overall result the most (words with only a small effect are not shown).")
        against = words[words["effect"] <= -0.2].head(6)["word"].tolist()
        for_it = words[words["effect"] >= 0.2].tail(6)["word"].tolist()[::-1]
        if not against and not for_it:
            st.markdown("No single word had a strong effect.")
        if against:
            st.markdown("Pushing towards **not recommending**:<br>" + "".join('<span class="chip" style="background:#FBD9D0;">' + html.escape(w) + "</span>" for w in against), unsafe_allow_html=True)
        if for_it:
            st.markdown("Pushing towards **recommending**:<br>" + "".join('<span class="chip" style="background:#D5F0DE;">' + html.escape(w) + "</span>" for w in for_it), unsafe_allow_html=True)
    with right:
        st.subheader("💡 What could the airline do?")
        problems = [] if len(pieces) == 0 else [t for t in pieces[pieces["label"] == "negative"]["topic"].drop_duplicates()]
        if problems:
            for topic in problems:
                st.markdown("- **" + name_of(topic) + ":** " + SUGGESTIONS[topic])
        else:
            st.markdown("No problems found in this review. 🎉")

    with st.expander("How does this work?"):
        st.markdown("1. The review is split into small pieces (sentences and clauses).\n"
                    "2. Word lists decide which **topic** each piece is about (Notebook 5).\n"
                    "3. A model trained on 51,000 real reviews gives each piece a score from 0 (unhappy) to 1 (happy) (Notebooks 4 and 6).\n"
                    "4. The same model scores the whole review to estimate if the passenger recommends the airline.")


# ================================================================ page 6: about
def show_about():
    page_intro("ℹ️ About " + PROJECT_NAME, "**" + PROJECT_NAME + "** is an NLP project that reads thousands of airline reviews and tells airlines what their passengers are unhappy about, and what to fix first.")

    st.subheader("How it works")
    steps = [("1️⃣ Collect", "64,740 airline reviews from Skytrax (2015 to 2025)"),
             ("2️⃣ Clean", "remove empty, short and duplicate reviews, tidy the text"),
             ("3️⃣ Understand", "a model learns whether a review recommends the airline (about 94% accurate)"),
             ("4️⃣ Find topics", "13 topics such as delays, seats and food, found with POS tagging and word lists"),
             ("5️⃣ Advise", "find the biggest problems and suggest what to improve")]
    for column, (title, text) in zip(st.columns(5), steps):
        with column:
            with st.container(border=True):
                st.markdown("**" + title + "**")
                st.caption(text)

    left, right = st.columns(2)
    with left:
        st.subheader("What you can do here")
        st.markdown("- **Overview:** how passengers feel, by airline, class and year\n"
                    "- **Top Problems:** what hurts most\n"
                    "- **Compare Airlines:** put 2 to 4 airlines side by side\n"
                    "- **Airline Report Card:** one-page summary with advice\n"
                    "- **Review Analyser:** try the system on any review")
        st.subheader("Main methods")
        st.markdown("Text cleaning, tokenization, lemmatization, n-grams, TF-IDF, Naive Bayes, Logistic Regression, VADER, "
                    "POS tagging, WordNet, aspect based sentiment analysis, LexRank summarisation, rule-based recommendations.")
    with right:
        st.subheader("How reliable is it?")
        st.markdown("- The review model is correct for about **94 out of 100** unseen reviews.\n"
                    "- Topic sentiment was checked against the passengers' own star ratings on unseen reviews, and agrees in "
                    "about **9 out of 10** clear cases.\n"
                    "- Ground staff and value / fees are the least reliable topics.")
        st.subheader("Good to know")
        st.markdown("- Reviews are written by people who chose to write, so they are not a random sample of all passengers.\n"
                    "- The results show **patterns**, not proof of cause.\n"
                    "- The suggested actions are general ideas. The data decides which ones are shown.")


# ================================================================ main
def main():
    st.markdown(STYLE, unsafe_allow_html=True)

    missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
    if missing:
        st.error("Some files are missing. Please run Notebooks 1 to 10 first, and start the app from the project folder.")
        st.write(missing)
        st.stop()

    st.sidebar.markdown("## ✈️ " + PROJECT_NAME)
    st.sidebar.caption("Airline passenger feedback analyser")
    page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed", key="page")
    st.sidebar.divider()
    st.sidebar.caption("Data: Skytrax airline reviews (2015 to 2025), analysed with NLP.")

    banner(TAGLINE)

    {"🏠 Overview": show_overview, "🚨 Top Problems": show_problems, "⚖️ Compare Airlines": show_compare,
     "📋 Airline Report Card": show_report_card, "📝 Review Analyser": show_analyser, "ℹ️ About SkyVoice": show_about}[page]()


if __name__ == "__main__":
    main()
