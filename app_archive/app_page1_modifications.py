import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
from datetime import date, timedelta

st.set_page_config(page_title="Ecommerce Analytics Command Center", layout="wide")

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem;}
    div[data-testid="metric-container"] {
        background-color: #f7f7f8;
        border: 1px solid #e6e6e6;
        padding: 14px;
        border-radius: 12px;
    }
    .business-box {
        background-color: #fafafa;
        border-left: 5px solid #222;
        padding: 16px 18px;
        border-radius: 8px;
        margin: 12px 0 22px 0;
    }
    .decision-box {
        background-color: #fff8e8;
        border-left: 5px solid #d99a00;
        padding: 16px 18px;
        border-radius: 8px;
        margin: 12px 0 22px 0;
    }
    .insight-box {
        background-color: #eef6ff;
        border-left: 5px solid #3b82f6;
        padding: 16px 18px;
        border-radius: 8px;
        margin: 12px 0 22px 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data loading & preparation
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"

    # If app.py is still inside a folder like "1. Python/", use the parent data folder.
    if not DATA_DIR.exists():
        DATA_DIR = BASE_DIR.parent / "data"

    customers_df = pd.read_csv(DATA_DIR / "customers_data.csv")
    orders_df = pd.read_csv(DATA_DIR / "orders_data.csv")
    products_df = pd.read_csv(DATA_DIR / "products_data.csv")
    sessions_df = pd.read_csv(DATA_DIR / "sessions_data.csv")
    seg_table = pd.read_csv(DATA_DIR / "Segmentation Summary.csv")
    return customers_df, orders_df, products_df, sessions_df, seg_table


def prepare_data(customers_df, orders_df, products_df, sessions_df):
    customers_df = customers_df.copy()
    orders_df = orders_df.copy()
    products_df = products_df.copy()
    sessions_df = sessions_df.copy()

    product_info_cols = [c for c in ['product_id', 'category', 'subcategory', 'price', 'cost', 'Stocking_Date', 'Inventory_Batch'] if c in products_df.columns]
    if 'Inventory_Batch' in products_df.columns:
        products_dedup = products_df[products_df['Inventory_Batch'] == 1][product_info_cols].copy()
        products_dedup = products_dedup.drop(columns=['Inventory_Batch'], errors='ignore')
    else:
        products_dedup = products_df[product_info_cols].drop_duplicates('product_id')

    orders_df = orders_df.merge(products_dedup, on="product_id", how="left", suffixes=("", "_product"))
    orders_df = orders_df.merge(customers_df, on="customer_id", how="left", suffixes=("", "_customer"))

    orders_df["order_date"] = pd.to_datetime(orders_df["order_date"], errors="coerce")
    if "signup_date" in customers_df.columns:
        customers_df["signup_date"] = pd.to_datetime(customers_df["signup_date"], errors="coerce")
    if "timestamp" in sessions_df.columns:
        sessions_df["timestamp"] = pd.to_datetime(sessions_df["timestamp"], errors="coerce")

    return customers_df, orders_df, products_df, sessions_df

customers_df, orders_df, products_df, sessions_df, seg_table = load_data()
customers_df, orders_df, products_df, sessions_df = prepare_data(customers_df, orders_df, products_df, sessions_df)

# -----------------------------------------------------------------------------
# Sidebar navigation and global filters
# -----------------------------------------------------------------------------
st.sidebar.title("📊 Command Center")
page = st.sidebar.radio(
    "Business Question",
    [
        "1. Executive Overview",
        "2. Revenue Drivers & Growth Strategy",
        "3. Marketing Investment",
        "4. Store Relocation Scenario",
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Global filters")

categories = ["All"] + sorted(orders_df["category"].dropna().unique().tolist()) if "category" in orders_df.columns else ["All"]
selected_category = st.sidebar.radio("Category", categories, index=0)

if selected_category != "All" and "category" in orders_df.columns:
    filtered_orders = orders_df[orders_df["category"] == selected_category].copy()
else:
    filtered_orders = orders_df.copy()

if "subcategory" in filtered_orders.columns:
    subcategories = sorted(filtered_orders["subcategory"].dropna().unique().tolist())
    selected_subcategories = st.sidebar.multiselect("Subcategory", subcategories, default=subcategories)
    if selected_subcategories:
        filtered_orders = filtered_orders[filtered_orders["subcategory"].isin(selected_subcategories)]

min_date = filtered_orders["order_date"].min().date() if not filtered_orders.empty else date(2019, 1, 1)
max_date = filtered_orders["order_date"].max().date() if not filtered_orders.empty else date(2024, 12, 31)
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_orders = filtered_orders[
        (filtered_orders["order_date"] >= pd.to_datetime(start_date)) &
        (filtered_orders["order_date"] <= pd.to_datetime(end_date))
    ]

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def money(value):
    return f"${value:,.0f}"


def pct(value):
    return f"{value:.1%}" if pd.notnull(value) else "N/A"


def get_total_metrics(df):
    total_revenue = df["price"].sum() if "price" in df.columns else 0
    total_orders = df["order_id"].nunique() if "order_id" in df.columns else 0
    active_customers = df["customer_id"].nunique() if "customer_id" in df.columns else 0
    returned_orders = df[df.get("status", "") == "Returned"]["order_id"].nunique() if "status" in df.columns else 0
    return_rate = returned_orders / total_orders if total_orders else 0
    aov = total_revenue / total_orders if total_orders else 0
    return total_revenue, total_orders, active_customers, return_rate, aov


def make_revenue_trend(df):
    trend = df.copy()
    trend["order_month"] = trend["order_date"].dt.to_period("M").astype(str)
    trend["category"] = trend["category"].fillna("Unknown")
    sales_trend = trend.groupby(["order_month", "category"], as_index=False)["price"].sum()

    fig = px.area(
        sales_trend,
        x="order_month", y="price", color="category",
        title="Revenue Trend by Category",
        markers=True, line_shape="spline",
        labels={"order_month": "Month", "price": "Revenue", "category": "Category"}
    )

    total_revenue_trend = sales_trend.groupby("order_month", as_index=False)["price"].sum()
    if len(total_revenue_trend) >= 2:
        total_revenue_trend["numeric_date"] = pd.to_datetime(total_revenue_trend["order_month"]).astype("int64") // 10**9
        model = LinearRegression()
        X = total_revenue_trend[["numeric_date"]]
        y = total_revenue_trend["price"]
        model.fit(X, y)
        total_revenue_trend["trend"] = model.predict(X)
        fig.add_trace(go.Scatter(
            x=total_revenue_trend["order_month"],
            y=total_revenue_trend["trend"],
            mode="lines",
            name="Total Revenue Trend",
            line=dict(color="black", width=3, dash="dash")
        ))
    return fig


def make_category_sunburst(df):
    cat_subcat_rev = df.groupby(["category", "subcategory"], as_index=False)["price"].sum()
    fig = px.sunburst(
        cat_subcat_rev,
        path=["category", "subcategory"], values="price",
        title="Revenue Mix by Category and Subcategory",
        labels={"price": "Revenue"}
    )
    fig.update_layout(margin=dict(t=60, l=0, r=0, b=0))
    return fig

def make_monthly_revenue_chart(df):
    monthly = df.copy()
    monthly["order_month"] = monthly["order_date"].dt.to_period("M").dt.to_timestamp()
    trend = monthly.groupby("order_month", as_index=False)["price"].sum()
    fig = px.line(
        trend,
        x="order_month", y="price",
        markers=True,
        title="Monthly Revenue Trend",
        labels={"order_month": "Month", "price": "Revenue"}
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(yaxis_tickprefix="$", hovermode="x unified")
    return fig


def make_monthly_orders_chart(df):
    monthly = df.copy()
    monthly["order_month"] = monthly["order_date"].dt.to_period("M").dt.to_timestamp()
    trend = monthly.groupby("order_month", as_index=False)["order_id"].nunique()
    trend.rename(columns={"order_id": "orders"}, inplace=True)
    fig = px.line(
        trend,
        x="order_month", y="orders",
        markers=True,
        title="Monthly Orders Trend",
        labels={"order_month": "Month", "orders": "Orders"}
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(hovermode="x unified")
    return fig


def make_monthly_aov_chart(df):
    monthly = df.copy()
    monthly["order_month"] = monthly["order_date"].dt.to_period("M").dt.to_timestamp()
    trend = monthly.groupby("order_month").agg(
        revenue=("price", "sum"),
        orders=("order_id", "nunique")
    ).reset_index()
    trend["aov"] = np.where(trend["orders"] > 0, trend["revenue"] / trend["orders"], 0)
    fig = px.line(
        trend,
        x="order_month", y="aov",
        markers=True,
        title="Monthly Average Order Value Trend",
        labels={"order_month": "Month", "aov": "Average Order Value"}
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(yaxis_tickprefix="$", hovermode="x unified")
    return fig


def get_location_group(df):
    location_col = None
    for possible_col in ["is_local", "state", "location", "customer_state"]:
        if possible_col in df.columns:
            location_col = possible_col
            break

    result = df.copy()
    if location_col == "is_local":
        result["customer_region"] = np.where(result["is_local"].astype(bool), "Local / MA", "Non-local")
    elif location_col in ["state", "customer_state", "location"]:
        result["customer_region"] = np.where(
            result[location_col].astype(str).str.upper().str.contains("MA|MASSACHUSETTS", na=False),
            "Local / MA", "Non-local"
        )
    else:
        result["customer_region"] = "Unknown"
    return result


def make_share_bar(df, dimension, title, top_n=None):
    summary = df.groupby(dimension, as_index=False)["price"].sum().sort_values("price", ascending=False)
    if top_n and len(summary) > top_n:
        top = summary.head(top_n).copy()
        other_value = summary.iloc[top_n:]["price"].sum()
        if other_value > 0:
            other = pd.DataFrame({dimension: ["Other"], "price": [other_value]})
            summary = pd.concat([top, other], ignore_index=True)
    summary["share"] = summary["price"] / summary["price"].sum() if summary["price"].sum() else 0
    fig = px.bar(
        summary,
        x=dimension, y="price", text="share",
        title=title,
        labels={dimension: dimension.replace("_", " ").title(), "price": "Revenue", "share": "Revenue Share"}
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(yaxis_tickprefix="$")
    return fig, summary


def concentration_metric(df, dimension, top_n=1):
    if dimension not in df.columns or df.empty:
        return np.nan
    summary = df.groupby(dimension)["price"].sum().sort_values(ascending=False)
    total = summary.sum()
    if total == 0:
        return np.nan
    return summary.head(top_n).sum() / total


def diagnose_owner_priority(df, sessions):
    """Directional executive signal for Page 1. This is intentionally simple and explainable."""
    revenue_share_top_category = concentration_metric(df, "category", top_n=1)
    top5_subcat_share = concentration_metric(df, "subcategory", top_n=5)

    return_rate = 0
    if "status" in df.columns and df["order_id"].nunique() > 0:
        return_rate = df[df["status"].eq("Returned")]["order_id"].nunique() / df["order_id"].nunique()

    conversion_rate = np.nan
    if "converted" in sessions.columns:
        conversion_rate = sessions["converted"].astype(bool).mean()

    # Simple rule-based logic that is easy to explain in interviews.
    if pd.notnull(conversion_rate) and conversion_rate < 0.03:
        priority = "Traffic acquisition / conversion quality"
        rationale = "Conversion appears relatively weak, so the owner should investigate traffic source quality before simply increasing product assortment."
    elif pd.notnull(revenue_share_top_category) and revenue_share_top_category > 0.55:
        priority = "Operational simplification"
        rationale = "Revenue is heavily concentrated in one category, so the owner should evaluate whether a narrower operating model could preserve revenue while reducing complexity."
    elif return_rate > 0.08:
        priority = "Margin improvement"
        rationale = "Return activity creates margin pressure, so the owner should investigate categories with higher return risk before pursuing aggressive growth."
    elif pd.notnull(top5_subcat_share) and top5_subcat_share > 0.70:
        priority = "Revenue concentration management"
        rationale = "A small number of subcategories drive most sales, so leadership should protect these winners while testing adjacent growth opportunities."
    else:
        priority = "Revenue growth"
        rationale = "No single risk signal dominates the overview, so the owner can focus on growth while continuing to monitor concentration and channel quality."

    return priority, rationale


def make_session_trend(sessions):
    sessions = sessions.copy()
    sessions["session_month"] = sessions["timestamp"].dt.to_period("M").astype(str)
    session_trend = sessions.groupby(["session_month", "source"], as_index=False)["session_id"].count()
    session_trend.rename(columns={"session_id": "sessions"}, inplace=True)
    fig = px.area(
        session_trend,
        x="session_month", y="sessions", color="source",
        title="Traffic Trend by Marketing Source",
        markers=True, line_shape="spline",
        labels={"session_month": "Month", "sessions": "Sessions", "source": "Source"}
    )
    return fig


def make_funnel(sessions):
    sessions = sessions.copy()
    sessions["converted"] = sessions["converted"].astype(bool)
    sessions["has_cart"] = sessions["page_views"] >= 3
    sessions["checkout_started"] = sessions["page_views"] >= 6
    funnel_counts = {
        "Sessions": len(sessions),
        "Add to Cart": sessions[sessions["has_cart"]].shape[0],
        "Checkout Started": sessions[sessions["checkout_started"]].shape[0],
        "Order Placed": sessions[sessions["converted"]].shape[0]
    }
    fig = go.Figure(go.Funnel(
        y=list(funnel_counts.keys()),
        x=list(funnel_counts.values()),
        textinfo="value+percent initial"
    ))
    fig.update_layout(title="Website Conversion Funnel")
    return fig


def make_segment_bubble(seg_table):
    seg = seg_table.copy()
    seg["text_label"] = seg["cluster"].astype(str) + "<br>N=" + seg["count"].astype(str)
    fig = px.scatter(
        seg,
        x="mean_tenure",
        y="mean_recency_days",
        size="sum_customer_lifetime_value",
        size_max=80,
        color="mean_high_value_order_ratio",
        color_continuous_scale="RdBu_r",
        text="text_label",
        title="Customer Segments for Campaign Planning",
        labels={
            "mean_tenure": "Tenure (days)",
            "mean_recency_days": "Recency (days)",
            "mean_high_value_order_ratio": "High-Value Order Ratio"
        }
    )
    positions = np.where(seg["mean_recency_days"] < 120, "top center", "bottom center")
    fig.update_traces(textposition=positions, marker=dict(opacity=0.72))
    fig.add_annotation(
        text="Bubble size = total CLV; color = high-value purchase ratio",
        xref="paper", yref="paper", x=0.5, y=1.12,
        showarrow=False, font=dict(size=13)
    )
    return fig


def source_performance_table(sessions):
    sessions = sessions.copy()
    sessions["converted"] = sessions["converted"].astype(bool)
    source_perf = sessions.groupby("source").agg(
        sessions=("session_id", "count"),
        avg_page_views=("page_views", "mean"),
        conversions=("converted", "sum")
    ).reset_index()
    source_perf["conversion_rate"] = source_perf["conversions"] / source_perf["sessions"]
    return source_perf.sort_values("conversion_rate", ascending=False)


def generate_campaign_suggestions(seg_table_df):
    from openai import OpenAI
    import holidays

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
    upcoming = {d: name for d, name in us_holidays.items() if today <= d <= today + timedelta(days=45)}
    holiday_context = "\n".join(f"- {name} on {d.strftime('%B %d')}" for d, name in sorted(upcoming.items())) or "No major U.S. holidays in the next 45 days."

    seasonal_context = {
        "Spring": "Spring refresh, Mother's Day, graduation season",
        "Summer": "Summer styling, wedding season, travel, Pride Month",
        "Fall": "Fall layering, back-to-school, early holiday preview",
        "Winter": "Holiday gifting, New Year reset, winter sales"
    }

    prompt = f"""
You are an ecommerce marketing strategist for a boutique fashion and accessories retailer.

Business objective: recommend practical marketing actions across Instagram, Facebook, and Google/SEO using the customer segment summary below.

Today: {today.strftime('%B %d, %Y')}
Seasonal context: {seasonal_context[season]}
Upcoming U.S. holidays: {holiday_context}

Please provide:
1. A concise persona summary for each segment.
2. Recommended channel focus: Instagram, Facebook, Google/SEO, or Email.
3. Campaign idea and message angle.
4. Whether the segment should receive discount, exclusivity, education/content, or reactivation messaging.

Use a strategic, executive-ready tone. Be concise.

Segment summary:
{seg_table_df.to_string(index=False)}
"""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a retail marketing strategist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

# -----------------------------------------------------------------------------
# PAGE 1: Executive Overview
# -----------------------------------------------------------------------------
if page == "1. Executive Overview":
    st.title("Executive Overview")
    st.markdown("""
    <div class="business-box">
    <b>Business question:</b> Is the business healthy, and where should leadership focus first?<br><br>
    This page acts as the owner's diagnostic overview. It does not make the final strategy decision;
    it identifies whether the next conversation should focus on revenue growth, margin improvement,
    traffic acquisition, or operational simplification.
    </div>
    """, unsafe_allow_html=True)

    # Page-level filters are intentionally placed here so the owner can test whether
    # the business health pattern holds across the whole business or only within
    # specific categories/subcategories.
    st.subheader("View filters")
    st.caption("Use these filters to test whether revenue, orders, AOV, and concentration patterns are broad-based or category-specific.")

    page1_orders = orders_df.copy()
    all_page1_categories = sorted(page1_orders["category"].dropna().unique().tolist()) if "category" in page1_orders.columns else []
    selected_page1_categories = st.multiselect(
        "Category",
        all_page1_categories,
        default=all_page1_categories,
        help="Question answered: Is the performance pattern broad-based, or driven by a specific category?"
    )
    if selected_page1_categories and "category" in page1_orders.columns:
        page1_orders = page1_orders[page1_orders["category"].isin(selected_page1_categories)].copy()

    all_page1_subcategories = sorted(page1_orders["subcategory"].dropna().unique().tolist()) if "subcategory" in page1_orders.columns else []
    selected_page1_subcategories = st.multiselect(
        "Subcategory",
        all_page1_subcategories,
        default=all_page1_subcategories,
        help="Question answered: Are business trends concentrated in only a few product types?"
    )
    if selected_page1_subcategories and "subcategory" in page1_orders.columns:
        page1_orders = page1_orders[page1_orders["subcategory"].isin(selected_page1_subcategories)].copy()

    p1_min_date = page1_orders["order_date"].min().date() if not page1_orders.empty else date(2019, 1, 1)
    p1_max_date = page1_orders["order_date"].max().date() if not page1_orders.empty else date(2024, 12, 31)
    p1_date_range = st.date_input("Date range for executive overview", [p1_min_date, p1_max_date], key="page1_date_range")
    if len(p1_date_range) == 2:
        p1_start_date, p1_end_date = p1_date_range
        page1_orders = page1_orders[
            (page1_orders["order_date"] >= pd.to_datetime(p1_start_date)) &
            (page1_orders["order_date"] <= pd.to_datetime(p1_end_date))
        ].copy()

    st.markdown("---")

    total_revenue, total_orders, active_customers, return_rate, aov = get_total_metrics(page1_orders)
    total_customers = customers_df["customer_id"].nunique() if "customer_id" in customers_df.columns else active_customers
    conversion_rate = sessions_df["converted"].astype(bool).mean() if "converted" in sessions_df.columns else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", money(total_revenue))
    c2.metric("Total Orders", f"{total_orders:,}")
    c3.metric("Active Customers", f"{active_customers:,}")
    c4.metric("Avg Order Value", money(aov))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Customers", f"{total_customers:,}")
    c6.metric("Conversion Rate", pct(conversion_rate))
    c7.metric("Return Rate", pct(return_rate))
    c8.metric("Data Through", str(page1_orders["order_date"].max().date()) if not page1_orders.empty else "N/A")

    st.subheader("1. Business health trend")
    st.markdown("""
    <div class="insight-box">
    <b>Question answered:</b> Is performance improving because the business is getting more orders, because customers are spending more per order, or both?<br><br>
    Reading revenue, orders, and AOV together helps distinguish volume-driven growth from price/order-value-driven growth.
    </div>
    """, unsafe_allow_html=True)

    if page1_orders.empty:
        st.warning("No orders match the selected filters.")
    # else:
    #     t1, t2, t3 = st.columns(3)
    #     with t1:
    #         st.plotly_chart(make_monthly_revenue_chart(page1_orders), use_container_width=True)
    #     with t2:
    #         st.plotly_chart(make_monthly_orders_chart(page1_orders), use_container_width=True)
    #     with t3:
    #         st.plotly_chart(make_monthly_aov_chart(page1_orders), use_container_width=True)
    else:
        st.plotly_chart(make_monthly_revenue_chart(page1_orders), use_container_width=True)
        st.plotly_chart(make_monthly_orders_chart(page1_orders), use_container_width=True)
        st.plotly_chart(make_monthly_aov_chart(page1_orders), use_container_width=True)
        
    st.subheader("2. Revenue concentration risk")
    st.markdown("""
    <div class="insight-box">
    <b>Question answered:</b> Is the business becoming too dependent on a few categories, product types, or local customers?<br><br>
    Concentration is not automatically bad. It can reveal winners. But it also creates risk if growth relies on a narrow part of the business.
    </div>
    """, unsafe_allow_html=True)

    page1_geo_orders = get_location_group(page1_orders)
    top_category_share = concentration_metric(page1_orders, "category", top_n=1)
    top5_subcat_share = concentration_metric(page1_orders, "subcategory", top_n=5)
    local_revenue_share = concentration_metric(page1_geo_orders, "customer_region", top_n=1)
    top10_product_share = concentration_metric(page1_orders, "product_id", top_n=10)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Top Category Share", pct(top_category_share) if pd.notnull(top_category_share) else "N/A")
    m2.metric("Top 5 Subcategory Share", pct(top5_subcat_share) if pd.notnull(top5_subcat_share) else "N/A")
    m3.metric("Largest Region Share", pct(local_revenue_share) if pd.notnull(local_revenue_share) else "N/A")
    m4.metric("Top 10 Product Share", pct(top10_product_share) if pd.notnull(top10_product_share) else "N/A")

    c1, c2 = st.columns(2)
    with c1:
        if "category" in page1_orders.columns and not page1_orders.empty:
            fig_cat_share, _ = make_share_bar(page1_orders, "category", "Revenue Share by Category")
            st.plotly_chart(fig_cat_share, use_container_width=True)
    with c2:
        if "subcategory" in page1_orders.columns and not page1_orders.empty:
            fig_subcat_share, _ = make_share_bar(page1_orders, "subcategory", "Revenue Share by Top Subcategories", top_n=8)
            st.plotly_chart(fig_subcat_share, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if not page1_geo_orders.empty:
            fig_geo_share, _ = make_share_bar(page1_geo_orders, "customer_region", "Revenue Share by Geography")
            st.plotly_chart(fig_geo_share, use_container_width=True)
    with c4:
        if "product_id" in page1_orders.columns and not page1_orders.empty:
            fig_product_share, _ = make_share_bar(page1_orders, "product_id", "Revenue Share by Top Products", top_n=10)
            st.plotly_chart(fig_product_share, use_container_width=True)

    st.subheader("3. Owner priority diagnosis")
    priority, rationale = diagnose_owner_priority(page1_orders, sessions_df)
    st.markdown(f"""
    <div class="decision-box">
    <b>Directional priority:</b> {priority}<br><br>
    <b>Why:</b> {rationale}<br><br>
    <b>How to use this:</b> This is not an automated final decision. It is an executive signal that tells the owner which deeper page to open next:
    revenue strategy, marketing investment, or operational simplification.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="business-box">
    <b>Page 1 decision logic:</b>
    <ul>
        <li>If revenue and orders are weak, prioritize <b>revenue growth</b>.</li>
        <li>If traffic converts poorly, prioritize <b>traffic acquisition and conversion quality</b>.</li>
        <li>If return pressure is high, prioritize <b>margin improvement</b>.</li>
        <li>If revenue is concentrated in complex categories, prioritize <b>operational simplification</b>.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 2: Revenue Drivers & Growth Strategy
# -----------------------------------------------------------------------------
elif page == "2. Revenue Drivers & Growth Strategy":
    st.title("Revenue Drivers & Growth Strategy")
    st.markdown("""
    <div class="business-box">
    <b>Business question:</b> Where does revenue come from, and what growth strategy should the business pursue?<br><br>
    This page combines revenue driver analysis with the owner's assortment strategy question:
    should the business continue supporting apparel, or shift focus toward handbags, accessories, and jewelry-style products?
    </div>
    """, unsafe_allow_html=True)

    category_summary = filtered_orders.groupby("category").agg(
        revenue=("price", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_id", "nunique"),
        avg_price=("price", "mean")
    ).reset_index().sort_values("revenue", ascending=False)
    category_summary["revenue_share"] = category_summary["revenue"] / category_summary["revenue"].sum()

    if "cost" in filtered_orders.columns:
        category_summary["gross_profit"] = filtered_orders.groupby("category").apply(lambda x: (x["price"] - x["cost"]).sum()).values
        category_summary["gross_margin"] = category_summary["gross_profit"] / category_summary["revenue"]

    if "status" in filtered_orders.columns:
        returns = filtered_orders.assign(is_returned=filtered_orders["status"].eq("Returned"))
        return_summary = returns.groupby("category")["is_returned"].mean().reset_index(name="return_rate")
        category_summary = category_summary.merge(return_summary, on="category", how="left")

    st.subheader("Revenue concentration by category")
    col1, col2 = st.columns([1.25, 1])
    with col1:
        fig_bar = px.bar(
            category_summary,
            x="category", y="revenue", text="revenue_share",
            title="Revenue Contribution by Category",
            labels={"revenue": "Revenue", "category": "Category", "revenue_share": "Revenue Share"}
        )
        fig_bar.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        display_cols = ["category", "revenue", "revenue_share", "orders", "avg_price"]
        if "gross_margin" in category_summary.columns:
            display_cols.append("gross_margin")
        if "return_rate" in category_summary.columns:
            display_cols.append("return_rate")
        st.dataframe(
            category_summary[display_cols].style.format({
                "revenue": "${:,.0f}",
                "revenue_share": "{:.1%}",
                "avg_price": "${:,.0f}",
                "gross_margin": "{:.1%}",
                "return_rate": "{:.1%}"
            }),
            use_container_width=True
        )

    st.subheader("Category growth over time")
    st.plotly_chart(make_revenue_trend(filtered_orders), use_container_width=True)

    st.subheader("Assortment strategy: apparel vs. non-apparel")
    apparel_keywords = ["Clothing", "Apparel"]
    apparel_subcats = ["Cardigan", "Pants", "Shirts & Tops", "Coats & Jackets", "Skirts"]
    assortment = filtered_orders.copy()
    assortment["assortment_group"] = np.where(
        assortment["category"].isin(apparel_keywords) | assortment["subcategory"].isin(apparel_subcats),
        "Apparel",
        "Handbags / Accessories / Jewelry"
    )

    assortment_summary = assortment.groupby("assortment_group").agg(
        revenue=("price", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_id", "nunique"),
        product_count=("product_id", "nunique"),
        avg_price=("price", "mean")
    ).reset_index()
    assortment_summary["revenue_share"] = assortment_summary["revenue"] / assortment_summary["revenue"].sum()
    if "cost" in assortment.columns:
        margin_summary = assortment.groupby("assortment_group").apply(lambda x: (x["price"] - x["cost"]).sum()).reset_index(name="gross_profit")
        assortment_summary = assortment_summary.merge(margin_summary, on="assortment_group", how="left")
        assortment_summary["gross_margin"] = assortment_summary["gross_profit"] / assortment_summary["revenue"]
    if "status" in assortment.columns:
        ret = assortment.assign(is_returned=assortment["status"].eq("Returned"))
        assortment_summary = assortment_summary.merge(ret.groupby("assortment_group")["is_returned"].mean().reset_index(name="return_rate"), on="assortment_group", how="left")

    col1, col2 = st.columns([1, 1])
    with col1:
        fig_assort = px.bar(
            assortment_summary,
            x="assortment_group", y="revenue", text="revenue_share",
            title="Revenue Trade-Off: Apparel vs. Focused Assortment",
            labels={"assortment_group": "Assortment Group", "revenue": "Revenue"}
        )
        fig_assort.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        st.plotly_chart(fig_assort, use_container_width=True)
    with col2:
        st.dataframe(
            assortment_summary.style.format({
                "revenue": "${:,.0f}",
                "revenue_share": "{:.1%}",
                "avg_price": "${:,.0f}",
                "gross_profit": "${:,.0f}",
                "gross_margin": "{:.1%}",
                "return_rate": "{:.1%}"
            }),
            use_container_width=True
        )

    st.markdown("""
    <div class="decision-box">
    <b>Decision supported:</b> This page does not force a yes/no answer. It frames the trade-off.
    Apparel may contribute meaningful revenue, but it also creates operational burden through sizing, returns, photoshoots, and inventory complexity.
    A focused assortment strategy should be evaluated by comparing lost revenue against margin improvement and operating simplicity.
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 3: Marketing Investment
# -----------------------------------------------------------------------------
elif page == "3. Marketing Investment":
    st.title("Marketing Investment")
    st.markdown("""
    <div class="business-box">
    <b>Business question:</b> How should limited marketing resources be allocated across Instagram, Facebook, Google/SEO, and email-style campaigns?<br><br>
    The owner does not have unlimited time or budget. This page compares traffic behavior and conversion quality to support channel prioritization.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(make_session_trend(sessions_df), use_container_width=True)
    with col2:
        st.plotly_chart(make_funnel(sessions_df), use_container_width=True)

    st.subheader("Traffic quality by source")
    source_perf = source_performance_table(sessions_df)
    col1, col2 = st.columns([1.25, 1])
    with col1:
        fig_source = px.bar(
            source_perf,
            x="source", y="conversion_rate", text="conversion_rate",
            title="Conversion Rate by Traffic Source",
            labels={"source": "Traffic Source", "conversion_rate": "Conversion Rate"}
        )
        fig_source.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        st.plotly_chart(fig_source, use_container_width=True)
    with col2:
        st.dataframe(
            source_perf.style.format({
                "avg_page_views": "{:.1f}",
                "conversion_rate": "{:.1%}"
            }),
            use_container_width=True
        )

    st.subheader("Campaign planning by customer segment")
    st.plotly_chart(make_segment_bubble(seg_table), use_container_width=True)

    with st.expander("🤖 AI-assisted campaign planner"):
        st.markdown("Generate executive-ready campaign suggestions from the customer segment summary.")
        if st.button("Generate campaign recommendations"):
            if "OPENAI_API_KEY" not in st.secrets:
                st.warning("OPENAI_API_KEY is not configured in Streamlit Secrets.")
            else:
                with st.spinner("Generating campaign recommendations..."):
                    st.markdown(generate_campaign_suggestions(seg_table))

    st.markdown("""
    <div class="decision-box">
    <b>Decisions supported:</b>
    <ul>
        <li>Use high-converting channels for direct-response campaigns.</li>
        <li>Use visually driven channels such as Instagram for awareness, styling, and product storytelling.</li>
        <li>Use Google/SEO to capture customers already showing purchase intent.</li>
        <li>Use customer segments to avoid broad discounting and tailor campaign messages.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 4: Store Relocation Scenario
# -----------------------------------------------------------------------------
elif page == "4. Store Relocation Scenario":
    st.title("Store Relocation Scenario")
    st.markdown("""
    <div class="business-box">
    <b>Business question:</b> Should the business move from a stronger retail location to a lower-rent location?<br><br>
    The owner considered reducing rent, but the risk is losing walk-in and local customers who contribute meaningfully to revenue.
    This page frames the decision as a trade-off between rent savings and potential revenue loss.
    </div>
    """, unsafe_allow_html=True)

    scenario_df = filtered_orders.copy()
    location_col = None
    for possible_col in ["is_local", "state", "location", "customer_state"]:
        if possible_col in scenario_df.columns:
            location_col = possible_col
            break

    if location_col == "is_local":
        scenario_df["customer_group"] = np.where(scenario_df["is_local"].astype(bool), "Local / MA", "Non-local")
    elif location_col in ["state", "customer_state", "location"]:
        scenario_df["customer_group"] = np.where(scenario_df[location_col].astype(str).str.upper().str.contains("MA|MASSACHUSETTS"), "Local / MA", "Non-local")
    else:
        st.warning("No local/customer location field found. Add `is_local` or `state` to support relocation analysis.")
        scenario_df["customer_group"] = "Unknown"

    local_summary = scenario_df.groupby("customer_group").agg(
        revenue=("price", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_id", "nunique"),
        avg_order_value=("price", "mean")
    ).reset_index()
    local_summary["revenue_share"] = local_summary["revenue"] / local_summary["revenue"].sum()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig_local = px.pie(
            local_summary,
            names="customer_group", values="revenue",
            title="Revenue Dependence on Local Customers"
        )
        st.plotly_chart(fig_local, use_container_width=True)
    with col2:
        st.dataframe(
            local_summary.style.format({
                "revenue": "${:,.0f}",
                "avg_order_value": "${:,.0f}",
                "revenue_share": "{:.1%}"
            }),
            use_container_width=True
        )

    st.subheader("Rent-saving break-even scenario")
    total_revenue, _, _, _, _ = get_total_metrics(scenario_df)
    annual_rent_savings = st.slider("Estimated annual rent savings from relocation", 0, 300000, 60000, step=5000)
    potential_local_revenue_loss = st.slider("Potential loss of local revenue", 0, 100, 20, step=5) / 100

    local_revenue = local_summary.loc[local_summary["customer_group"].eq("Local / MA"), "revenue"].sum()
    estimated_revenue_loss = local_revenue * potential_local_revenue_loss
    net_effect = annual_rent_savings - estimated_revenue_loss

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated Rent Savings", money(annual_rent_savings))
    c2.metric("Estimated Local Revenue at Risk", money(estimated_revenue_loss))
    c3.metric("Estimated Net Effect", money(net_effect))

    scenario_chart = pd.DataFrame({
        "Component": ["Rent Savings", "Revenue at Risk", "Net Effect"],
        "Value": [annual_rent_savings, -estimated_revenue_loss, net_effect]
    })
    fig_scenario = px.bar(
        scenario_chart, x="Component", y="Value", text="Value",
        title="Scenario View: Rent Savings vs. Revenue Risk"
    )
    fig_scenario.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig_scenario, use_container_width=True)

    st.markdown("""
    <div class="decision-box">
    <b>Decision supported:</b> The dashboard does not decide relocation automatically. It gives the owner a break-even framework:
    relocation only makes sense if rent savings exceed the expected loss from walk-in/local revenue, plus any brand visibility loss that is harder to quantify.
    </div>
    """, unsafe_allow_html=True)
