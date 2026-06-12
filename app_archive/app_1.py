import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
from datetime import date, timedelta
import holidays
from openai import OpenAI

st.set_page_config(page_title="Ecommerce Analytics Command Center", layout="wide")

# -----------------------------
# Global styling
# -----------------------------
st.markdown(
    """
    <style>
        .section-card {
            background-color: #f7f7f9;
            padding: 1rem 1.2rem;
            border-radius: 12px;
            border: 1px solid #e6e6e6;
            margin-bottom: 1rem;
        }
        .decision-card {
            background-color: #fff8e8;
            padding: 1rem 1.2rem;
            border-radius: 12px;
            border-left: 5px solid #f0b429;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .muted {
            color: #666666;
            font-size: 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"

    customers_df = pd.read_csv(DATA_DIR / "customers_data.csv")
    orders_df = pd.read_csv(DATA_DIR / "orders_data.csv")
    products_df = pd.read_csv(DATA_DIR / "products_data.csv")
    sessions_df = pd.read_csv(DATA_DIR / "sessions_data.csv")
    seg_table = pd.read_csv(DATA_DIR / "Segmentation Summary.csv")

    return customers_df, orders_df, products_df, sessions_df, seg_table

customers_df, orders_df_raw, products_df, sessions_df_raw, seg_table = load_data()

# Prepare merged order-level data
product_info_cols = ["product_id", "category", "subcategory", "price", "cost", "Stocking_Date"]
products_df_dedup = products_df[products_df["Inventory_Batch"] == 1][product_info_cols]
orders_df = orders_df_raw.merge(products_df_dedup, on="product_id", how="left", suffixes=("", "_dup"))
orders_df = orders_df.merge(customers_df, on="customer_id", how="left").reset_index(drop=True)
orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])
customers_df["signup_date"] = pd.to_datetime(customers_df["signup_date"])
sessions_df = sessions_df_raw.copy()

# -----------------------------
# Sidebar navigation + filters
# -----------------------------
st.sidebar.title("📊 Command Center")
page = st.sidebar.radio(
    "Business Question",
    [
        "🏢 Executive Overview",
        "❤️ Customer Retention & Marketing",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

categories = ["All"] + sorted(list(orders_df["category"].dropna().unique()))
selected_category = st.sidebar.radio("Category", options=categories, index=0)

if selected_category == "All":
    subcategories = sorted(list(orders_df["subcategory"].dropna().unique()))
else:
    subcategories = sorted(list(orders_df.loc[orders_df["category"] == selected_category, "subcategory"].dropna().unique()))
selected_subcategory = st.sidebar.multiselect("Subcategory", options=subcategories, default=subcategories)

date_range = st.sidebar.date_input("Order Date Range", [])

filtered_orders = orders_df.copy()
if selected_category != "All":
    filtered_orders = filtered_orders[filtered_orders["category"] == selected_category]
if selected_subcategory:
    filtered_orders = filtered_orders[filtered_orders["subcategory"].isin(selected_subcategory)]
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_orders = filtered_orders[
        (filtered_orders["order_date"] >= pd.to_datetime(start_date))
        & (filtered_orders["order_date"] <= pd.to_datetime(end_date))
    ]

# -----------------------------
# Helper functions
# -----------------------------
def compute_sales_trend(df):
    df = df.copy()
    df["order_month"] = pd.to_datetime(df["order_date"], errors="coerce").dt.to_period("M").astype(str)
    df["category"] = df["category"].fillna("Miscellaneous")
    trend = df.groupby(["order_month", "category"])["price"].sum().reset_index()
    trend.rename(columns={"order_month": "order_date"}, inplace=True)
    return trend


def add_trend_line(fig, trend_df, date_col, value_col, name):
    total_trend = trend_df.groupby(date_col)[value_col].sum().reset_index()
    total_trend["numeric_date"] = pd.to_datetime(total_trend[date_col]).astype("int64") // 10**9
    model = LinearRegression()
    model.fit(total_trend[["numeric_date"]], total_trend[value_col])
    total_trend["trend"] = model.predict(total_trend[["numeric_date"]])

    fig.add_trace(
        go.Scatter(
            x=total_trend[date_col],
            y=total_trend["trend"],
            mode="lines",
            name=name,
            line=dict(color="black", width=3, dash="dash"),
        )
    )
    return fig


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0


def format_currency(value):
    return f"${value:,.0f}"


def get_seasonal_campaign_prompt(seg_table_df):
    today = date.today()
    month = today.month
    if 2 <= month <= 4:
        season = "Spring"
    elif 5 <= month <= 7:
        season = "Summer"
    elif 8 <= month <= 10:
        season = "Fall"
    else:
        season = "Winter"

    us_holidays = holidays.UnitedStates()
    upcoming = {
        d: name
        for d, name in us_holidays.items()
        if today <= d <= today + timedelta(days=45)
    }
    holiday_context = "\n".join(
        f"- {name} on {d.strftime('%B %d')}" for d, name in sorted(upcoming.items())
    ) or "No major U.S. holidays in the next 45 days."

    seasonal_occasion_context = {
        "Spring": "Spring refresh, Mother's Day, graduation season",
        "Summer": "Summer kick-off, wedding season, vacation styling",
        "Fall": "Back to school, fall layering, Halloween preview",
        "Winter": "Holiday gifting, New Year reset, winter sales",
    }

    prompt = f"""
You are a customer insights analyst and ecommerce marketing strategist for a fashion retail brand.

Today is {today.strftime('%B %d, %Y')}.
Current season: {season}
Seasonal retail context: {seasonal_occasion_context[season]}
Upcoming U.S. holidays: {holiday_context}

Use the customer segmentation summary below to generate an executive-ready marketing plan.

For each segment, provide:
1. A concise customer persona description.
2. The business risk or opportunity.
3. A recommended email campaign.
4. A recommended social or content campaign.
5. A clear action priority: High / Medium / Low.

Keep the tone polished, practical, and business-focused. Avoid sounding like a student project.

Segmentation summary:
{seg_table_df.to_string(index=False)}
"""
    return prompt


def generate_campaign_suggestions(seg_table_df):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a senior ecommerce customer insights analyst."},
            {"role": "user", "content": get_seasonal_campaign_prompt(seg_table_df)},
        ],
        temperature=0.6,
    )
    return response.choices[0].message.content

# -----------------------------
# Shared metric calculations
# -----------------------------
total_revenue = filtered_orders["price"].sum()
total_orders = filtered_orders["order_id"].nunique()
total_customers = customers_df.shape[0]
active_customers = filtered_orders["customer_id"].nunique()
returned_orders = filtered_orders.loc[filtered_orders["status"] == "Returned", "order_id"].nunique()
return_rate = safe_divide(returned_orders, total_orders)
aov = safe_divide(total_revenue, total_orders)

purchase_frequency = safe_divide(total_orders, total_customers)
customers_ltv_df = customers_df.copy()
customers_ltv_df["first_purchase"] = customers_ltv_df["customer_id"].map(filtered_orders.groupby("customer_id")["order_date"].min())
customers_ltv_df["last_purchase"] = customers_ltv_df["customer_id"].map(filtered_orders.groupby("customer_id")["order_date"].max())
customers_ltv_df["customer_lifespan"] = (
    customers_ltv_df["last_purchase"] - customers_ltv_df["first_purchase"]
).dt.days / 365
average_customer_lifespan = customers_ltv_df["customer_lifespan"].mean(skipna=True)
ltv = aov * purchase_frequency * (average_customer_lifespan if pd.notna(average_customer_lifespan) else 0)

# -----------------------------
# Page 1: Executive Overview
# -----------------------------
if page == "🏢 Executive Overview":
    st.title("🏢 Executive Overview")
    st.caption("Business problem: Leadership needs a single source of truth to monitor performance, understand growth drivers, and identify where action is needed.")

    st.markdown(
        """
        <div class="section-card">
        <b>Questions this page answers</b><br>
        • Is the business growing sustainably?<br>
        • Which product categories are driving revenue?<br>
        • Are traffic and conversion patterns supporting sales growth?<br>
        • Where should leadership investigate next?
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", format_currency(total_revenue))
    col2.metric("Total Orders", f"{total_orders:,}")
    col3.metric("Active Customers", f"{active_customers:,}")
    col4.metric("Avg Order Value", f"${aov:,.2f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Total Customers", f"{total_customers:,}")
    col6.metric("Return Rate", f"{return_rate:.2%}")
    col7.metric("Estimated LTV", f"${ltv:,.2f}")
    col8.metric("Conversion Rate", "1.20%")

    st.markdown("### 1. Revenue Growth and Category Mix")
    sales_trend = compute_sales_trend(filtered_orders)
    fig_sales = px.area(
        sales_trend,
        x="order_date",
        y="price",
        color="category",
        title="Revenue Breakdown Over Time",
        markers=True,
        line_shape="spline",
        template="plotly_dark",
        labels={"price": "Revenue", "order_date": "Month", "category": "Category"},
    )
    fig_sales = add_trend_line(fig_sales, sales_trend, "order_date", "price", "Total Revenue Trend")
    fig_sales.update_layout(title_font=dict(size=20))

    cat_subcat_rev = filtered_orders.groupby(["category", "subcategory"]).agg({"price": "sum"}).reset_index()
    fig_cat_subcat = px.sunburst(
        cat_subcat_rev,
        path=["category", "subcategory"],
        values="price",
        title="Revenue Breakdown by Category and Subcategory",
        template="plotly_white",
        color="category",
        labels={"price": "Revenue"},
        branchvalues="total",
    )
    fig_cat_subcat.update_layout(margin=dict(t=60, l=0, r=0, b=0), title_font=dict(size=20))

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig_sales, use_container_width=True)
    with col2:
        st.plotly_chart(fig_cat_subcat, use_container_width=True)

    st.markdown(
        """
        <div class="decision-card">
        <b>Decision supported:</b> Use category-level revenue trends to identify whether growth is broad-based or concentrated in a few product areas. If growth depends heavily on one category, leadership should monitor concentration risk and evaluate whether merchandising or promotion plans need to be diversified.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 2. Traffic Quality and Conversion Funnel")
    sessions_df["session_month"] = pd.to_datetime(sessions_df["timestamp"]).dt.to_period("M").astype(str)
    session_trend = sessions_df.groupby(["session_month", "source"])["session_id"].count().reset_index()
    session_trend.rename(columns={"session_month": "session_date", "session_id": "sessions"}, inplace=True)

    fig_sessions = px.area(
        session_trend,
        x="session_date",
        y="sessions",
        color="source",
        title="Session Trend by Traffic Source",
        markers=True,
        line_shape="spline",
        template="plotly_dark",
        labels={"sessions": "Session Count", "session_date": "Month", "source": "Traffic Source"},
    )
    fig_sessions = add_trend_line(fig_sessions, session_trend, "session_date", "sessions", "Total Session Trend")
    fig_sessions.update_layout(title_font=dict(size=20))

    sessions_df["converted"] = sessions_df["converted"].astype(bool)
    sessions_df["has_cart"] = sessions_df["page_views"] >= 3
    sessions_df["checkout_started"] = sessions_df["page_views"] >= 6
    funnel_counts = {
        "Sessions": len(sessions_df),
        "Add to Cart": sessions_df[sessions_df["has_cart"]].shape[0],
        "Checkout Started": sessions_df[sessions_df["checkout_started"]].shape[0],
        "Order Placed": sessions_df[sessions_df["converted"]].shape[0],
    }
    fig_funnel = go.Figure(
        go.Funnel(
            y=list(funnel_counts.keys()),
            x=list(funnel_counts.values()),
            textinfo="value+percent initial",
            marker={"color": ["#f8cb2e", "#bfbfbf", "#f49c6e", "#5ba8d0"]},
        )
    )
    fig_funnel.update_layout(title="Conversion Funnel Based on Session Data", title_font=dict(size=20))

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig_sessions, use_container_width=True)
    with col2:
        st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown(
        """
        <div class="decision-card">
        <b>Decision supported:</b> Compare traffic growth with funnel conversion to determine whether performance issues come from demand generation or onsite conversion. If sessions grow but orders do not, the business should investigate product pages, checkout friction, pricing, or inventory availability.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Page 2: Customer Retention & Marketing
# -----------------------------
elif page == "❤️ Customer Retention & Marketing":
    st.title("❤️ Customer Retention & Marketing")
    st.caption("Business problem: Marketing resources are limited, so the business needs to identify which customer groups deserve retention, reactivation, or VIP investment.")

    st.markdown(
        """
        <div class="section-card">
        <b>Questions this page answers</b><br>
        • Which customers drive the most value?<br>
        • Which customers show signs of churn risk?<br>
        • How should marketing messages differ by segment?<br>
        • How can AI accelerate campaign planning from analytical outputs?
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Customer Segment Value Map")
    seg_table = seg_table.copy()
    seg_table["text_label"] = seg_table["cluster"].astype(str) + "<br>N=" + seg_table["count"].astype(str)

    fig_segment = px.scatter(
        seg_table,
        x="mean_tenure",
        y="mean_recency_days",
        size="sum_customer_lifetime_value",
        size_max=80,
        color="mean_high_value_order_ratio",
        color_continuous_scale="RdBu_r",
        text="text_label",
        title="Customer Segmentation: Tenure vs. Recency",
        labels={
            "mean_tenure": "Tenure (days)",
            "mean_recency_days": "Recency (days since last purchase)",
            "mean_high_value_order_ratio": "High-Value Order Ratio",
            "sum_customer_lifetime_value": "Total CLV",
        },
    )
    positions = np.where(seg_table["mean_recency_days"] < 120, "top center", "bottom center")
    fig_segment.update_traces(
        textposition=positions,
        marker=dict(opacity=0.7),
        textfont=dict(size=12, color="black", family="Arial"),
    )
    fig_segment.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        title_font=dict(size=20),
        margin=dict(l=60, r=60, t=100, b=60),
        font=dict(family="Arial", size=16),
        xaxis=dict(showgrid=True, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridcolor="lightgray"),
    )
    fig_segment.add_annotation(
        text="Bubble size = Total CLV | Color = High-value order ratio | Higher recency = longer time since last purchase",
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.12,
        showarrow=False,
        font=dict(size=13),
        xanchor="center",
    )
    st.plotly_chart(fig_segment, use_container_width=True)

    st.markdown("### 2. Segment Summary Table")
    st.dataframe(seg_table.drop(columns=["text_label"], errors="ignore"), use_container_width=True)

    st.markdown(
        """
        <div class="decision-card">
        <b>Decision supported:</b> Segment customers by value and engagement level, then match each group to a different business action. High-value active customers should receive loyalty treatment, while high-recency customers should be prioritized for win-back campaigns.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 3. AI-Assisted Campaign Planning")
    st.markdown(
        """
        <div class="section-card">
        The AI assistant converts the segmentation output into executive-ready campaign ideas. This demonstrates how an analyst can use generative AI to speed up interpretation, messaging strategy, and stakeholder communication.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🤖 Generate segment-based campaign recommendations"):
        st.markdown("Click the button below to generate customer personas and campaign suggestions based on the current segment summary.")
        if st.button("Generate AI Campaign Plan"):
            if "OPENAI_API_KEY" not in st.secrets:
                st.error("OPENAI_API_KEY is not configured in Streamlit Secrets.")
            else:
                with st.spinner("Generating campaign recommendations..."):
                    gpt_response = generate_campaign_suggestions(seg_table.drop(columns=["text_label"], errors="ignore"))
                    st.markdown(gpt_response)
