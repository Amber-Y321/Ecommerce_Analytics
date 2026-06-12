import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Global date filter only. Category and subcategory filters are intentionally
# kept inside specific analysis sections so they do not accidentally hide
# whole-business risk signals.
filtered_orders = orders_df.copy()

min_date = orders_df["order_date"].min().date() if not orders_df.empty else date(2019, 1, 1)
max_date = orders_df["order_date"].max().date() if not orders_df.empty else date(2024, 12, 31)
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_orders = filtered_orders[
        (filtered_orders["order_date"] >= pd.to_datetime(start_date)) &
        (filtered_orders["order_date"] <= pd.to_datetime(end_date))
    ].copy()

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


def compact_trend_fig(fig, height=260):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=48, b=25),
        hovermode="x unified"
    )
    return fig


def make_orders_by_price_range(df):
    price_df = df.copy()
    if price_df.empty or "price" not in price_df.columns:
        return go.Figure()

    bins = [0, 300, 500, 800, 1200, np.inf]
    labels = ["<$300", "$300–$499", "$500–$799", "$800–$1,199", "$1,200+"]
    price_df["price_range"] = pd.cut(price_df["price"], bins=bins, labels=labels, right=False)
    summary = price_df.groupby("price_range", observed=False).agg(
        orders=("order_id", "nunique"),
        revenue=("price", "sum")
    ).reset_index()

    fig = px.bar(
        summary,
        x="price_range", y="orders", text="orders",
        title="Order Demand by Price Range",
        labels={"price_range": "Price Range", "orders": "Orders"},
        hover_data={"revenue": ":$,.0f"}
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=35))
    return fig


def make_traffic_conversion_chart(sessions):
    sessions = sessions.copy()
    if sessions.empty or "timestamp" not in sessions.columns:
        return go.Figure()

    sessions["session_month"] = sessions["timestamp"].dt.to_period("M").dt.to_timestamp()
    if "converted" in sessions.columns:
        sessions["converted_bool"] = sessions["converted"].astype(bool)
    else:
        sessions["converted_bool"] = False

    monthly = sessions.groupby("session_month").agg(
        sessions=("session_id", "count"),
        conversions=("converted_bool", "sum")
    ).reset_index()
    monthly["conversion_rate"] = np.where(monthly["sessions"] > 0, monthly["conversions"] / monthly["sessions"], 0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=monthly["session_month"], y=monthly["sessions"], name="Sessions"),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["session_month"],
            y=monthly["conversion_rate"],
            name="Conversion Rate",
            mode="lines+markers"
        ),
        secondary_y=True
    )
    fig.update_layout(
        title="Traffic and Conversion Health",
        hovermode="x unified",
        height=360,
        margin=dict(l=20, r=20, t=60, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Sessions", secondary_y=False)
    fig.update_yaxes(title_text="Conversion Rate", tickformat=".1%", secondary_y=True)
    return fig


def make_return_rate_by_category_chart(df):
    ret_df = df.copy()
    if ret_df.empty or "status" not in ret_df.columns or "category" not in ret_df.columns:
        return go.Figure()

    ret_df["is_returned"] = ret_df["status"].eq("Returned")
    summary = ret_df.groupby("category").agg(
        orders=("order_id", "nunique"),
        returned_orders=("is_returned", "sum")
    ).reset_index()
    summary["return_rate"] = np.where(summary["orders"] > 0, summary["returned_orders"] / summary["orders"], 0)
    summary = summary.sort_values("return_rate", ascending=False)

    fig = px.bar(
        summary,
        x="category", y="return_rate", text="return_rate",
        title="Return Rate by Category",
        labels={"category": "Category", "return_rate": "Return Rate"}
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(yaxis_tickformat=".1%", margin=dict(l=20, r=20, t=60, b=35))
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


def make_funnel(sessions, title="Website Conversion Funnel"):
    sessions = sessions.copy()
    if sessions.empty:
        funnel_counts = {"Sessions": 0, "Add to Cart": 0, "Checkout Started": 0, "Order Placed": 0}
    else:
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
    fig.update_layout(title=title)
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


def source_conversion_comparison_table(sessions):
    """Create source-level funnel conversion metrics for channel comparison.

    Since this simulated dataset does not contain actual funnel events, page_view
    thresholds are used as behavioral proxies:
    - page_views >= 3: Add to Cart proxy
    - page_views >= 6: Checkout Started proxy
    - converted == True: Order Placed
    """
    sessions = sessions.copy()
    if sessions.empty or "source" not in sessions.columns:
        return pd.DataFrame(columns=[
            "source", "sessions", "add_to_cart", "checkout_started", "orders",
            "add_to_cart_rate", "checkout_start_rate", "order_conversion_rate"
        ])

    sessions["converted"] = sessions["converted"].astype(bool)
    sessions["has_cart"] = sessions["page_views"] >= 3
    sessions["checkout_started"] = sessions["page_views"] >= 6

    comp = sessions.groupby("source").agg(
        sessions=("session_id", "count"),
        add_to_cart=("has_cart", "sum"),
        checkout_started=("checkout_started", "sum"),
        orders=("converted", "sum"),
        avg_page_views=("page_views", "mean")
    ).reset_index()

    comp["add_to_cart_rate"] = np.where(comp["sessions"] > 0, comp["add_to_cart"] / comp["sessions"], 0)
    comp["checkout_start_rate"] = np.where(comp["sessions"] > 0, comp["checkout_started"] / comp["sessions"], 0)
    comp["order_conversion_rate"] = np.where(comp["sessions"] > 0, comp["orders"] / comp["sessions"], 0)
    return comp.sort_values("order_conversion_rate", ascending=False)


def make_source_conversion_comparison_chart(source_conversion_df):
    """Grouped bar chart comparing funnel conversion rates across marketing sources."""
    if source_conversion_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Conversion Comparison by Marketing Source")
        return fig

    chart_df = source_conversion_df[[
        "source", "add_to_cart_rate", "checkout_start_rate", "order_conversion_rate"
    ]].copy()

    chart_df = chart_df.melt(
        id_vars="source",
        value_vars=["add_to_cart_rate", "checkout_start_rate", "order_conversion_rate"],
        var_name="stage",
        value_name="rate"
    )

    stage_labels = {
        "add_to_cart_rate": "Add-to-cart rate",
        "checkout_start_rate": "Checkout-start rate",
        "order_conversion_rate": "Order conversion rate"
    }
    chart_df["stage"] = chart_df["stage"].map(stage_labels)

    fig = px.bar(
        chart_df,
        x="source",
        y="rate",
        color="stage",
        barmode="group",
        text="rate",
        title="Conversion Comparison by Marketing Source",
        labels={"source": "Marketing Source", "rate": "Conversion Rate", "stage": "Funnel Stage"},
        template="plotly_white"
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(yaxis_tickformat=".1%", legend_title_text="Funnel Stage")
    return fig


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


def generate_marketing_ai_response(seg_table_df, source_perf_df, selected_question):
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

    prompt = f"""
You are an ecommerce marketing analyst for a boutique fashion, handbags, jewelry, and accessories retailer.

Business objective: help the owner decide how to allocate limited marketing time and budget across Instagram, Facebook, Google/SEO, and retention-style campaigns.

Selected business question:
{selected_question}

Current traffic source performance:
{source_perf_df.to_string(index=False)}

Customer segment summary:
{seg_table_df.to_string(index=False)}

Seasonal context:
- Today: {today.strftime('%B %d, %Y')}
- Upcoming U.S. holidays in the next 45 days:
{holiday_context}

Please answer in an executive-ready format:
1. Direct recommendation.
2. Evidence from traffic/conversion and customer segments.
3. Suggested channel allocation across Instagram, Facebook, Google/SEO, and retention campaigns.
4. Specific campaign ideas.
5. Risks or metrics to monitor next.

Be practical, concise, and avoid generic advice.
"""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a practical retail marketing strategist and analytics translator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
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
    This page is an executive diagnostic. It helps the owner quickly identify whether the next conversation should focus on
    revenue growth, traffic acquisition, margin improvement, or operational simplification.
    </div>
    """, unsafe_allow_html=True)

    # Page 1 uses the global date scope, while category/subcategory filters
    # below apply only to the business health trend section.
    overview_orders = filtered_orders.copy()
    total_revenue, total_orders, active_customers, return_rate, aov = get_total_metrics(overview_orders)
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
    c8.metric("Data Through", str(overview_orders["order_date"].max().date()) if not overview_orders.empty else "N/A")

    st.subheader("1. Business health trend")
    st.markdown("""
    <div class="insight-box">
    <b>Question answered:</b> Is performance changing because the business is getting more orders, customers are spending more per order, or both?<br><br>
    Category and subcategory filters in this section apply only to the three trend charts below. The global sidebar date range applies to the whole dashboard.
    </div>
    """, unsafe_allow_html=True)

    trend_orders = filtered_orders.copy()
    f1, f2 = st.columns([1, 1.2])
    with f1:
        all_categories = sorted(trend_orders["category"].dropna().unique().tolist()) if "category" in trend_orders.columns else []
        selected_categories = st.multiselect(
            "Category",
            all_categories,
            default=all_categories,
            help="Use this to test whether business health trends are broad-based or category-specific."
        )
    if selected_categories and "category" in trend_orders.columns:
        trend_orders = trend_orders[trend_orders["category"].isin(selected_categories)].copy()

    with f2:
        all_subcategories = sorted(trend_orders["subcategory"].dropna().unique().tolist()) if "subcategory" in trend_orders.columns else []
        selected_subcategories = st.multiselect(
            "Subcategory",
            all_subcategories,
            default=all_subcategories,
            help="Use this to see whether a small set of product types is driving the trend."
        )
    if selected_subcategories and "subcategory" in trend_orders.columns:
        trend_orders = trend_orders[trend_orders["subcategory"].isin(selected_subcategories)].copy()


    if trend_orders.empty:
        st.warning("No orders match the selected trend filters.")
    else:
        st.plotly_chart(compact_trend_fig(make_monthly_revenue_chart(trend_orders)), use_container_width=True)
        st.plotly_chart(compact_trend_fig(make_monthly_orders_chart(trend_orders)), use_container_width=True)
        st.plotly_chart(compact_trend_fig(make_monthly_aov_chart(trend_orders)), use_container_width=True)

    st.subheader("2. Revenue concentration risk")
    st.markdown("""
    <div class="insight-box">
    <b>Question answered:</b> Is the business becoming too dependent on a few categories, product types, local customers, or price tiers?<br><br>
    Concentration is not automatically bad. It can reveal winners. But it also creates risk if growth relies on a narrow part of the business.
    </div>
    """, unsafe_allow_html=True)

    concentration_orders = orders_df.copy()
    geo_orders = get_location_group(concentration_orders)
    top_category_share = concentration_metric(concentration_orders, "category", top_n=1)
    top5_subcat_share = concentration_metric(concentration_orders, "subcategory", top_n=5)
    largest_region_share = concentration_metric(geo_orders, "customer_region", top_n=1)

    m1, m2, m3 = st.columns(3)
    m1.metric("Top Category Share", pct(top_category_share) if pd.notnull(top_category_share) else "N/A")
    m2.metric("Top 5 Subcategory Share", pct(top5_subcat_share) if pd.notnull(top5_subcat_share) else "N/A")
    m3.metric("Largest Region Share", pct(largest_region_share) if pd.notnull(largest_region_share) else "N/A")

    c1, c2 = st.columns(2)
    with c1:
        if "category" in concentration_orders.columns and not concentration_orders.empty:
            fig_cat_share, _ = make_share_bar(concentration_orders, "category", "Revenue Share by Category")
            st.plotly_chart(fig_cat_share, use_container_width=True)
    with c2:
        if "subcategory" in concentration_orders.columns and not concentration_orders.empty:
            fig_subcat_share, _ = make_share_bar(concentration_orders, "subcategory", "Revenue Share by Top Subcategories", top_n=8)
            st.plotly_chart(fig_subcat_share, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if not geo_orders.empty:
            fig_geo_share, _ = make_share_bar(geo_orders, "customer_region", "Revenue Share by Geography")
            st.plotly_chart(fig_geo_share, use_container_width=True)
    with c4:
        st.plotly_chart(make_orders_by_price_range(concentration_orders), use_container_width=True)

    st.markdown("""
    <div class="business-box">
    <b>Why price range is useful here:</b> It shows whether demand is concentrated in entry, mid, or premium price tiers.
    This helps the owner understand whether revenue growth depends on higher-priced items or a broader base of accessible purchases.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("3. Traffic, conversion, and margin alerts")
    st.markdown("""
    <div class="insight-box">
    <b>Question answered:</b> Are there warning signs that should shift the owner's attention from pure revenue growth to traffic quality or margin protection?<br><br>
    Traffic and conversion show whether demand generation is healthy. Return rate highlights potential margin pressure and operational friction.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.35, 1])
    with col1:
        st.plotly_chart(make_traffic_conversion_chart(sessions_df), use_container_width=True)
    with col2:
        st.plotly_chart(make_return_rate_by_category_chart(overview_orders), use_container_width=True)

    return_threshold = 0.08
    conversion_threshold = 0.03
    if return_rate >= return_threshold:
        st.error(f"High return rate alert: overall return rate is {return_rate:.1%}. The owner should investigate category-level return drivers before pushing aggressive revenue growth.")
    else:
        st.success(f"Return rate check: overall return rate is {return_rate:.1%}, below the {return_threshold:.0%} alert threshold.")

    if pd.notnull(conversion_rate) and conversion_rate < conversion_threshold:
        st.warning(f"Conversion alert: overall conversion rate is {conversion_rate:.1%}. The owner should review traffic source quality and website funnel performance before increasing marketing spend.")
    elif pd.notnull(conversion_rate):
        st.info(f"Conversion check: overall conversion rate is {conversion_rate:.1%}. Use the Marketing Investment page to compare traffic source quality.")

    priority, rationale = diagnose_owner_priority(overview_orders, sessions_df)
    st.markdown(f"""
    <div class="decision-box">
    <b>Directional priority:</b> {priority}<br><br>
    <b>Why:</b> {rationale}<br><br>
    <b>How to use this:</b> This overview is not an automated final decision. It tells the owner which deeper page to open next:
    revenue strategy, marketing investment, or operational simplification.
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 2: Revenue Drivers & Growth Strategy
# -----------------------------------------------------------------------------
elif page == "2. Revenue Drivers & Growth Strategy":
    st.title("Revenue Drivers & Growth Strategy")
    st.markdown("""
    <div class="business-box">
    <b>Business question:</b> Which product categories should the owner bet on, reduce, or test next?<br><br>
    This page combines <b>current revenue drivers</b>, <b>future growth signals</b>, <b>profitability</b>, and <b>operational effort</b> to support a realistic assortment strategy.
    The owner's goal is not simply to sell more, but to gradually pivot from lower-profit, high-effort product lines toward categories that can generate better profit with less operational burden.
    </div>
    """, unsafe_allow_html=True)

    page2_orders = filtered_orders.copy()

    if page2_orders.empty:
        st.warning("No orders available for the selected date range.")
    else:
        # ------------------------------------------------------------------
        # Assumptions and profit preparation
        # ------------------------------------------------------------------
        page2_orders = page2_orders.copy()
        page2_orders["category"] = page2_orders["category"].fillna("Unknown")
        page2_orders["subcategory"] = page2_orders["subcategory"].fillna("Unknown")

        # Use actual price and cost if available. If cost is missing, profit-related charts will still run with 0.
        if "cost" in page2_orders.columns:
            page2_orders["gross_profit"] = page2_orders["price"] - page2_orders["cost"]
        else:
            page2_orders["gross_profit"] = 0

        # Operational effort assumptions are explicit and editable.
        # Higher score = harder to manage because of sizing, photoshoots, sourcing complexity, shipping/handling, and return risk.
        effort_assumptions = {
            "Cardigan": 4.0,
            "Pants": 4.5,
            "Shirts & Tops": 4.5,
            "Coats & Jackets": 5.0,
            "Skirts": 4.0,
            "Handbags": 2.5,
            "Bracelets": 1.5,
            "Charms & Pendants": 1.5,
            "Scarf": 2.0,
        }
        page2_orders["effort_score"] = page2_orders["subcategory"].map(effort_assumptions).fillna(3.0)

        # ------------------------------------------------------------------
        # Build subcategory and category driver summaries
        # ------------------------------------------------------------------
        def build_driver_summary(df, dims):
            base = (
                df.groupby(dims, as_index=False)
                .agg(
                    revenue=("price", "sum"),
                    gross_profit=("gross_profit", "sum"),
                    orders=("order_id", "nunique"),
                    customers=("customer_id", "nunique"),
                    avg_price=("price", "mean"),
                    avg_effort=("effort_score", "mean"),
                    product_count=("product_id", "nunique")
                )
                .sort_values("revenue", ascending=False)
            )
            total_revenue = base["revenue"].sum()
            total_profit = base["gross_profit"].sum()
            base["revenue_share"] = np.where(total_revenue > 0, base["revenue"] / total_revenue, 0)
            base["profit_share"] = np.where(total_profit > 0, base["gross_profit"] / total_profit, 0)
            base["gross_margin"] = np.where(base["revenue"] > 0, base["gross_profit"] / base["revenue"], 0)

            if "status" in df.columns:
                ret = df.assign(is_returned=df["status"].eq("Returned"))
                ret_summary = (
                    ret.groupby(dims, as_index=False)
                    .agg(return_rate=("is_returned", "mean"))
                )
                base = base.merge(ret_summary, on=dims, how="left")
            else:
                base["return_rate"] = 0

            return base

        category_summary = build_driver_summary(page2_orders, ["category"])
        subcat_summary = build_driver_summary(page2_orders, ["category", "subcategory"])

        # ------------------------------------------------------------------
        # Recent trend signal: compare recent 6 months with prior 6 months.
        # This helps separate current drivers from possible future drivers.
        # ------------------------------------------------------------------
        max_order_date = page2_orders["order_date"].max()
        recent_start = max_order_date - pd.DateOffset(months=6)
        previous_start = max_order_date - pd.DateOffset(months=12)

        page2_orders["period_signal"] = np.select(
            [
                page2_orders["order_date"] > recent_start,
                (page2_orders["order_date"] > previous_start) & (page2_orders["order_date"] <= recent_start),
            ],
            ["Recent 6M", "Previous 6M"],
            default="Earlier"
        )

        def add_growth_signal(summary_df, dims):
            period = (
                page2_orders[page2_orders["period_signal"].isin(["Recent 6M", "Previous 6M"])]
                .groupby(dims + ["period_signal"], as_index=False)
                .agg(period_revenue=("price", "sum"), period_profit=("gross_profit", "sum"), period_orders=("order_id", "nunique"))
            )
            if period.empty:
                summary_df["recent_revenue"] = 0
                summary_df["previous_revenue"] = 0
                summary_df["recent_growth"] = 0
                return summary_df

            pivot = period.pivot_table(
                index=dims,
                columns="period_signal",
                values="period_revenue",
                aggfunc="sum",
                fill_value=0
            ).reset_index()
            if "Recent 6M" not in pivot.columns:
                pivot["Recent 6M"] = 0
            if "Previous 6M" not in pivot.columns:
                pivot["Previous 6M"] = 0

            pivot = pivot.rename(columns={"Recent 6M": "recent_revenue", "Previous 6M": "previous_revenue"})
            pivot["recent_growth"] = np.where(
                pivot["previous_revenue"] > 0,
                (pivot["recent_revenue"] - pivot["previous_revenue"]) / pivot["previous_revenue"],
                np.where(pivot["recent_revenue"] > 0, 1, 0)
            )
            return summary_df.merge(pivot[dims + ["recent_revenue", "previous_revenue", "recent_growth"]], on=dims, how="left").fillna({
                "recent_revenue": 0,
                "previous_revenue": 0,
                "recent_growth": 0,
            })

        category_summary = add_growth_signal(category_summary, ["category"])
        subcat_summary = add_growth_signal(subcat_summary, ["category", "subcategory"])

        def classify_driver(row):
            high_current = row["revenue_share"] >= 0.15
            growing = row["recent_growth"] >= 0.15
            declining = row["recent_growth"] <= -0.15
            high_profit = row["gross_margin"] >= subcat_summary["gross_margin"].median()
            low_effort = row["avg_effort"] <= 2.5

            if high_current and growing and high_profit:
                return "Core future driver"
            if high_current and declining:
                return "Current driver under watch"
            if (not high_current) and growing and high_profit:
                return "Emerging future driver"
            if high_current:
                return "Current revenue driver"
            if high_profit and low_effort:
                return "Profit-efficient test"
            return "Lower priority / monitor"

        subcat_summary["driver_signal"] = subcat_summary.apply(classify_driver, axis=1)

        # Strategy recommendation at subcategory level.
        def recommend_action(row):
            if row["gross_margin"] >= 0.45 and row["recent_growth"] >= 0.10 and row["avg_effort"] <= 2.5:
                return "Expand / invest"
            if row["revenue_share"] >= 0.12 and row["avg_effort"] >= 4.0:
                return "Maintain selectively"
            if row["gross_margin"] >= 0.45 and row["avg_effort"] <= 3.0:
                return "Test & grow"
            if row["recent_growth"] < -0.20 and row["gross_margin"] < 0.35:
                return "Reduce exposure"
            return "Monitor"

        subcat_summary["suggested_action"] = subcat_summary.apply(recommend_action, axis=1)

        # ------------------------------------------------------------------
        # 1. Current drivers: revenue and profit mix
        # ------------------------------------------------------------------
        st.subheader("1. Current drivers: what carries the business today?")
        st.markdown("""
        <div class="insight-box">
        <b>Question answered:</b> Which product areas currently generate revenue and gross profit?<br><br>
        This distinguishes products that are simply popular from products that may actually support a stronger business model.
        </div>
        """, unsafe_allow_html=True)

        cat_subcat_rev = (
            page2_orders.groupby(["category", "subcategory"], as_index=False)
            .agg(price=("price", "sum"), gross_profit=("gross_profit", "sum"), orders=("order_id", "nunique"))
        )
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
        st.plotly_chart(fig_cat_subcat, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_rev_cat = px.bar(
                category_summary.sort_values("revenue", ascending=True),
                x="revenue",
                y="category",
                orientation="h",
                text="revenue_share",
                title="Revenue Share by Category",
                labels={"revenue": "Revenue", "category": "Category", "revenue_share": "Revenue Share"},
                template="plotly_white"
            )
            fig_rev_cat.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig_rev_cat.update_layout(xaxis_tickprefix="$", margin=dict(t=60, l=20, r=20, b=40))
            st.plotly_chart(fig_rev_cat, use_container_width=True)

        with col2:
            top_subcats = subcat_summary.sort_values("revenue", ascending=False).head(10).sort_values("revenue", ascending=True)
            fig_rev_subcat = px.bar(
                top_subcats,
                x="revenue",
                y="subcategory",
                orientation="h",
                color="category",
                text="revenue_share",
                title="Top Subcategories by Revenue",
                labels={"revenue": "Revenue", "subcategory": "Subcategory", "revenue_share": "Revenue Share"},
                template="plotly_white"
            )
            fig_rev_subcat.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig_rev_subcat.update_layout(xaxis_tickprefix="$", margin=dict(t=60, l=20, r=20, b=40))
            st.plotly_chart(fig_rev_subcat, use_container_width=True)

        # ------------------------------------------------------------------
        # 2. Current driver + evolving trend: temporary or future-facing?
        # ------------------------------------------------------------------
        st.subheader("2. Current driver + evolving trend")
        st.markdown("""
        <div class="insight-box">
        <b>Question answered:</b> Are today's revenue drivers temporary, seasonal, declining, or potential future drivers?<br><br>
        Current revenue share shows what matters today. Recent 6-month growth shows whether that driver is strengthening or weakening.
        </div>
        """, unsafe_allow_html=True)

        trend_df = page2_orders.copy()
        trend_df["order_month"] = trend_df["order_date"].dt.to_period("M").dt.to_timestamp()

        tab_cat, tab_subcat, tab_signal = st.tabs(["Category trend", "Subcategory trend", "Driver signal table"])

        with tab_cat:
            category_trend = (
                trend_df.groupby(["order_month", "category"], as_index=False)
                .agg(revenue=("price", "sum"), gross_profit=("gross_profit", "sum"))
            )
            fig_category_trend = px.line(
                category_trend,
                x="order_month",
                y="revenue",
                color="category",
                markers=True,
                title="Monthly Revenue Trend by Category",
                labels={"order_month": "Month", "revenue": "Revenue", "category": "Category"},
                template="plotly_white"
            )
            fig_category_trend.update_layout(yaxis_tickprefix="$", hovermode="x unified", margin=dict(t=60, l=20, r=20, b=40))
            st.plotly_chart(fig_category_trend, use_container_width=True)

        with tab_subcat:
            selected_top_subcats = subcat_summary.sort_values("revenue", ascending=False).head(8)["subcategory"].tolist()
            subcategory_trend = (
                trend_df[trend_df["subcategory"].isin(selected_top_subcats)]
                .groupby(["order_month", "subcategory"], as_index=False)
                .agg(revenue=("price", "sum"))
            )
            fig_subcategory_trend = px.line(
                subcategory_trend,
                x="order_month",
                y="revenue",
                color="subcategory",
                markers=True,
                title="Monthly Revenue Trend by Top Subcategories",
                labels={"order_month": "Month", "revenue": "Revenue", "subcategory": "Subcategory"},
                template="plotly_white"
            )
            fig_subcategory_trend.update_layout(yaxis_tickprefix="$", hovermode="x unified", margin=dict(t=60, l=20, r=20, b=40))
            st.plotly_chart(fig_subcategory_trend, use_container_width=True)

        with tab_signal:
            signal_cols = [
                "category", "subcategory", "revenue", "revenue_share", "gross_profit", "gross_margin",
                "orders", "recent_growth", "avg_effort", "return_rate", "driver_signal", "suggested_action"
            ]
            signal_display = subcat_summary[signal_cols].sort_values(["suggested_action", "revenue"], ascending=[True, False])
            st.dataframe(
                signal_display.style.format({
                    "revenue": "${:,.0f}",
                    "revenue_share": "{:.1%}",
                    "gross_profit": "${:,.0f}",
                    "gross_margin": "{:.1%}",
                    "recent_growth": "{:.1%}",
                    "avg_effort": "{:.1f}",
                    "return_rate": "{:.1%}",
                }),
                use_container_width=True,
                hide_index=True
            )

        # ------------------------------------------------------------------
        # 3. Profit-to-effort tradeoff
        # ------------------------------------------------------------------
        st.subheader("3. Profit-to-effort tradeoff")
        st.markdown("""
        <div class="insight-box">
        <b>Question answered:</b> Which categories make good money without creating too much operational burden?<br><br>
        This reflects the owner's concern that apparel may create more burden because of sizing, European sourcing/shipping, returns, and styling work, while jewelry and accessories may offer a better profit-to-effort profile.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1.2, 1])
        with col1:
            fig_profit_bubble = px.scatter(
                subcat_summary,
                x="gross_margin",
                y="revenue",
                size="orders",
                color="category",
                hover_name="subcategory",
                text="subcategory",
                title="Revenue vs Gross Margin by Subcategory",
                labels={
                    "gross_margin": "Gross Margin",
                    "revenue": "Revenue",
                    "orders": "Orders",
                    "category": "Category"
                },
                template="plotly_white",
                size_max=50
            )
            fig_profit_bubble.update_traces(textposition="top center")
            fig_profit_bubble.update_layout(
                xaxis_tickformat=".0%",
                yaxis_tickprefix="$",
                margin=dict(t=60, l=20, r=20, b=40)
            )
            st.plotly_chart(fig_profit_bubble, use_container_width=True)

        with col2:
            fig_effort_matrix = px.scatter(
                subcat_summary,
                x="avg_effort",
                y="gross_profit",
                size="revenue",
                color="suggested_action",
                hover_name="subcategory",
                text="subcategory",
                title="Assortment Strategy Matrix",
                labels={
                    "avg_effort": "Operational Effort Score",
                    "gross_profit": "Gross Profit",
                    "revenue": "Revenue",
                    "suggested_action": "Suggested Action"
                },
                template="plotly_white",
                size_max=50
            )
            median_effort = subcat_summary["avg_effort"].median()
            median_profit = subcat_summary["gross_profit"].median()
            fig_effort_matrix.add_vline(x=median_effort, line_dash="dash", line_color="gray")
            fig_effort_matrix.add_hline(y=median_profit, line_dash="dash", line_color="gray")
            fig_effort_matrix.update_traces(textposition="top center")
            fig_effort_matrix.update_layout(
                yaxis_tickprefix="$",
                margin=dict(t=60, l=20, r=20, b=40)
            )
            st.plotly_chart(fig_effort_matrix, use_container_width=True)

        st.markdown("""
        <div class="decision-box">
        <b>How to read this:</b><br>
        <ul>
            <li><b>High margin + low effort:</b> good candidates to expand or test more aggressively.</li>
            <li><b>High revenue + high effort:</b> important, but should be maintained selectively rather than expanded blindly.</li>
            <li><b>Low margin + high effort:</b> candidates for reduction or tighter buying discipline.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # 4. Assortment scenario builder
        # ------------------------------------------------------------------
        st.subheader("4. Assortment scenario: what if the owner pivots gradually?")
        st.markdown("""
        <div class="insight-box">
        <b>Question answered:</b> If the owner wants to pivot from lower-profit, high-effort products into higher-profit categories, what mix could preserve enough revenue while improving profit and simplicity?<br><br>
        The default scenario uses a middle-ground focus: jewelry, handbags, scarves, and selected cardigans. This can be changed below to test other assortment possibilities.
        </div>
        """, unsafe_allow_html=True)

        default_focus = ["Handbags", "Bracelets", "Charms & Pendants", "Scarf", "Cardigan"]
        all_subcats = sorted(subcat_summary["subcategory"].dropna().unique().tolist())
        selected_focus = st.multiselect(
            "Select subcategories to include in the focused assortment scenario",
            all_subcats,
            default=[s for s in default_focus if s in all_subcats]
        )

        scenario = page2_orders.copy()
        scenario["scenario_group"] = np.where(
            scenario["subcategory"].isin(selected_focus),
            "Focused assortment scenario",
            "Outside focused scenario"
        )

        scenario_summary = (
            scenario.groupby("scenario_group", as_index=False)
            .agg(
                revenue=("price", "sum"),
                gross_profit=("gross_profit", "sum"),
                orders=("order_id", "nunique"),
                customers=("customer_id", "nunique"),
                product_count=("product_id", "nunique"),
                avg_effort=("effort_score", "mean")
            )
        )
        scenario_summary["revenue_share"] = np.where(scenario_summary["revenue"].sum() > 0, scenario_summary["revenue"] / scenario_summary["revenue"].sum(), 0)
        scenario_summary["profit_share"] = np.where(scenario_summary["gross_profit"].sum() > 0, scenario_summary["gross_profit"] / scenario_summary["gross_profit"].sum(), 0)
        scenario_summary["gross_margin"] = np.where(scenario_summary["revenue"] > 0, scenario_summary["gross_profit"] / scenario_summary["revenue"], 0)

        if "status" in scenario.columns:
            scenario_ret = scenario.assign(is_returned=scenario["status"].eq("Returned"))
            scenario_ret_summary = scenario_ret.groupby("scenario_group", as_index=False).agg(return_rate=("is_returned", "mean"))
            scenario_summary = scenario_summary.merge(scenario_ret_summary, on="scenario_group", how="left")

        c1, c2 = st.columns([1, 1])
        with c1:
            fig_scenario = px.bar(
                scenario_summary,
                x="scenario_group",
                y=["revenue", "gross_profit"],
                barmode="group",
                title="Revenue and Gross Profit by Assortment Scenario",
                labels={"scenario_group": "Scenario", "value": "Amount", "variable": "Metric"},
                template="plotly_white"
            )
            fig_scenario.update_layout(yaxis_tickprefix="$", margin=dict(t=60, l=20, r=20, b=70))
            st.plotly_chart(fig_scenario, use_container_width=True)

        with c2:
            st.dataframe(
                scenario_summary.style.format({
                    "revenue": "${:,.0f}",
                    "gross_profit": "${:,.0f}",
                    "revenue_share": "{:.1%}",
                    "profit_share": "{:.1%}",
                    "gross_margin": "{:.1%}",
                    "avg_effort": "{:.1f}",
                    "return_rate": "{:.1%}",
                }),
                use_container_width=True,
                hide_index=True
            )

        # Timing matters: show whether selected scenario is gaining or losing share over time.
        timing = scenario.copy()
        timing["quarter"] = timing["order_date"].dt.to_period("Q").astype(str)
        timing_summary = (
            timing.groupby(["quarter", "scenario_group"], as_index=False)
            .agg(revenue=("price", "sum"), gross_profit=("gross_profit", "sum"))
        )
        quarter_total = timing_summary.groupby("quarter")["revenue"].transform("sum")
        timing_summary["revenue_share"] = np.where(quarter_total > 0, timing_summary["revenue"] / quarter_total, 0)

        fig_timing = px.line(
            timing_summary,
            x="quarter",
            y="revenue_share",
            color="scenario_group",
            markers=True,
            title="Focused Scenario Revenue Share Over Time",
            labels={"quarter": "Quarter", "revenue_share": "Revenue Share", "scenario_group": "Scenario"},
            template="plotly_white"
        )
        fig_timing.update_layout(yaxis_tickformat=".0%", hovermode="x unified", margin=dict(t=60, l=20, r=20, b=40))
        st.plotly_chart(fig_timing, use_container_width=True)

        st.markdown("""
        <div class="decision-box">
        <b>Recommended decision framing:</b> Do not immediately exit apparel. Test a gradual pivot toward jewelry, handbags, scarves, and selected cardigan styles while reducing exposure to lower-margin, high-effort apparel subcategories.<br><br>
        <b>Why timing matters:</b> If the focused scenario is gaining revenue share over recent quarters, the owner has stronger evidence to shift buying, marketing, and styling resources toward that mix. If the trend is seasonal or weakening, the safer choice is a slower test-and-learn approach.
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 3: Marketing Investment
# -----------------------------------------------------------------------------
elif page == "3. Marketing Investment":
    st.title("Marketing Investment")
    st.markdown("""
    <div class="business-box">
    <b>Business question:</b> How should limited marketing resources be allocated across Instagram, Facebook, Google/SEO, and retention-style campaigns?<br><br>
    The owner does not have unlimited time or budget. This page links traffic trends, source-specific funnel performance, and AI-assisted campaign planning so marketing decisions are based on quality of traffic, not just traffic volume.
    </div>
    """, unsafe_allow_html=True)

    # Apply global date scope to website sessions as well, so the marketing page
    # stays aligned with the dashboard's selected period.
    marketing_sessions = sessions_df.copy()
    if "timestamp" in marketing_sessions.columns and len(date_range) == 2:
        marketing_sessions = marketing_sessions[
            (marketing_sessions["timestamp"] >= pd.to_datetime(start_date)) &
            (marketing_sessions["timestamp"] <= pd.to_datetime(end_date))
        ].copy()

    st.subheader("1. Traffic source trend and source-specific conversion funnel")
    st.markdown("""
    <div class="insight-box">
    <b>Question answered:</b> Which marketing sources bring traffic, and does that traffic actually move through the website funnel?<br><br>
    Use the source selector below to compare all traffic or isolate channels such as Instagram, Facebook, Google, SEO, or direct traffic.
    The funnel updates with the selected source(s), so the owner can compare volume and conversion quality together.
    </div>
    """, unsafe_allow_html=True)

    all_sources = sorted(marketing_sessions["source"].dropna().unique().tolist()) if "source" in marketing_sessions.columns else []
    selected_sources = st.multiselect(
        "Select traffic source(s) to inspect",
        all_sources,
        default=all_sources,
        help="This selection controls both the traffic trend and the website conversion funnel below."
    )

    source_filtered_sessions = marketing_sessions.copy()
    if selected_sources and "source" in source_filtered_sessions.columns:
        source_filtered_sessions = source_filtered_sessions[source_filtered_sessions["source"].isin(selected_sources)].copy()

    selected_source_label = ", ".join(selected_sources) if selected_sources else "No source selected"

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(make_session_trend(source_filtered_sessions), use_container_width=True)
    with col2:
        st.plotly_chart(
            make_funnel(source_filtered_sessions, title=f"Website Conversion Funnel: {selected_source_label}"),
            use_container_width=True
        )

    st.subheader("2. Conversion comparison between marketing sources")
    st.markdown("""
    <div class="insight-box">
    <b>Question answered:</b> Which marketing source generates better conversion quality across the website funnel?<br><br>
    Traffic volume alone can be misleading. This comparison shows whether visitors from each source move from browsing to add-to-cart, checkout, and order placement.
    </div>
    """, unsafe_allow_html=True)

    source_perf = source_performance_table(marketing_sessions)
    source_conversion = source_conversion_comparison_table(marketing_sessions)

    st.plotly_chart(
        make_source_conversion_comparison_chart(source_conversion),
        use_container_width=True
    )

    st.dataframe(
        source_conversion.style.format({
            "avg_page_views": "{:.1f}",
            "add_to_cart_rate": "{:.1%}",
            "checkout_start_rate": "{:.1%}",
            "order_conversion_rate": "{:.1%}"
        }),
        use_container_width=True
    )

    st.markdown("""
    <div class="decision-box">
    <b>How to read this:</b>
    <ul>
        <li><b>High traffic + low order conversion:</b> useful for awareness, but needs better landing pages, offers, or targeting.</li>
        <li><b>Lower traffic + high order conversion:</b> a strong candidate for budget increase because the audience shows purchase intent.</li>
        <li><b>Strong add-to-cart but weak checkout:</b> review shipping, pricing, checkout friction, or trust signals.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("3. AI-assisted marketing decision support")
    st.plotly_chart(make_segment_bubble(seg_table), use_container_width=True)

    frequent_questions = [
        "Which channel should the owner prioritize next month: Instagram, Facebook, Google/SEO, or retention campaigns?",
        "Which traffic source brings the highest-quality visitors, not just the most visitors?",
        "Is Instagram working more as an awareness channel or a conversion channel?",
        "Should the owner invest more in Google/SEO or social media content?",
        "How should campaigns differ for high-value customers versus at-risk customers?",
        "What campaign should we run for jewelry, handbags, scarves, or selected cardigans?",
        "How can we improve conversion from website sessions to orders?",
        "What warning signs should the owner monitor before increasing marketing spend?"
    ]

    with st.expander("🤖 AI-assisted campaign planner", expanded=True):
        st.markdown("Choose a common owner question, or write your own. The AI uses the segment summary and traffic source performance shown on this page.")

        selected_question = st.selectbox(
            "Frequently asked marketing questions",
            frequent_questions,
            index=0
        )
        custom_question = st.text_area(
            "Optional: write your own question",
            placeholder="Example: If I only have time to focus on one channel this month, what should I do and why?"
        )
        final_question = custom_question.strip() if custom_question.strip() else selected_question

        if st.button("Generate AI recommendation"):
            if "OPENAI_API_KEY" not in st.secrets:
                st.warning("OPENAI_API_KEY is not configured in Streamlit Secrets.")
            else:
                with st.spinner("Generating marketing recommendation..."):
                    st.markdown(generate_marketing_ai_response(seg_table, source_perf, final_question))

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
