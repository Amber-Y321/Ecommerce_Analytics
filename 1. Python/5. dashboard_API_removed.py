import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Shopify Business Dashboard", layout="wide")


st.sidebar.title("📊 Dashboard Navigation")

# tab1, tab2 = st.tabs(["🏆 Business Overview", "🎯 Customer Segmentation"])
section = st.sidebar.radio("🧭 Go to Section", [
    "🏆 Business Overview",
    "🎯 Customer Segmentation"
])

# with tab1:
if section == "🏆 Business Overview":

    # Set dark theme
    st.markdown("""
        <style>
            body {
                background-color: #1e1e1e;
                color: white;
            }
            .sidebar .sidebar-content {
                background: #252526;
                width: 200px !important;
            }
            div[data-testid="metric-container"] {
                background-color: #222 !important; /* Darker black for better contrast */
                color: black !important;
                border-radius: 10px;
                padding: 10px;
                text-align: center;
                border: 1px solid #555 !important; /* Adds a slight border for better visibility */
            }
            div[data-testid="stMetricLabel"] {
                color: black !important;
                font-size: 16px;
            }
            div[data-testid="stMetricValue"] {
                color: black !important;
                font-weight: bold;
                font-size: 22px;
            }
            div[data-testid="stMetricDelta"] {
                color: #ffffff !important; /* Ensure delta values are visible */
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Load data
    @st.cache_data
    def load_data():
        customers_df = pd.read_csv("customers_data.csv")
        orders_df = pd.read_csv("orders_data.csv")
        products_df = pd.read_csv("products_data.csv")
        sessions_df = pd.read_csv("sessions_data.csv")
        return customers_df, orders_df, products_df, sessions_df
    
    customers_df, orders_df, products_df, sessions_df = load_data()
    
    # Deduplicate products before merging
    product_info_cols = ['product_id', 'category', 'subcategory', 'price', 'cost', 'Stocking_Date']
    products_df_dedup = products_df[products_df['Inventory_Batch'] == 1][product_info_cols]
    
    # Merge datasets
    orders_df = orders_df.merge(products_df_dedup, on="product_id", how="left", suffixes=("", "_dup"))
    orders_df = orders_df.merge(customers_df, on="customer_id", how="left").reset_index()
    
    # Convert dates
    orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])
    customers_df["signup_date"] = pd.to_datetime(customers_df["signup_date"])
    
    # Sidebar Filters    
    # Category Filter as Buttons
    categories = ["All"] + list(orders_df["category"].dropna().unique())
    selected_category = st.sidebar.radio("Select Category", options=categories, index=0)
    
    # Subcategory Filter as Buttons
    if selected_category == "All":
        subcategories = list(orders_df["subcategory"].dropna().unique()) 
    else:
        subcategories = list(orders_df[orders_df["category"]==selected_category]["subcategory"].dropna().unique())
    selected_subcategory = st.sidebar.multiselect("Select Subcategory", options=subcategories, default=subcategories)
    
    # Apply filters
    if selected_category != "All":
        orders_df = orders_df[orders_df["category"] == selected_category]
    if selected_subcategory:
        orders_df = orders_df[orders_df["subcategory"].isin(selected_subcategory)]
    
    date_range = st.sidebar.date_input("Select Date Range", [])
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (orders_df["order_date"] >= pd.to_datetime(start_date)) & (orders_df["order_date"] <= pd.to_datetime(end_date))
        orders_df = orders_df[mask]
        
    # Sales Metrics
    total_revenue = orders_df["price"].sum()
    total_orders = orders_df["order_id"].nunique()
    total_customers = customers_df.shape[0]
    
    ##  active customers
    active_customers = orders_df['customer_id'].nunique()
    
    # Return rate
        # Total Orders
    total_orders = orders_df["order_id"].nunique()
    returned_orders = orders_df[orders_df['status'] == 'Returned']["order_id"].nunique()
        # Compute Return Rate
    return_rate = returned_orders / total_orders
    
    # AOV
    AOV = round(total_revenue/ total_orders,2)
    
    # conversion rate
    conversion_rate = str(round(600 / 50000 *100,2) )+ '%'
    
    # LTV
    purchase_frequency = total_orders / total_customers 
    customers_df["first_purchase"] = customers_df["customer_id"].map(orders_df.groupby("customer_id")["order_date"].min())
    customers_df["last_purchase"] = customers_df["customer_id"].map(orders_df.groupby("customer_id")["order_date"].max())
    customers_df["customer_lifespan"] = (customers_df["last_purchase"] - customers_df["first_purchase"]).dt.days / 365  # Convert to years
    average_customer_lifespan = customers_df["customer_lifespan"].mean()
    LTV = AOV * purchase_frequency * average_customer_lifespan
    
    
    # Dashboard Layout
    st.markdown("<h3 style='text-align: center;'>🏆 Business Overview</h3>", unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
    with col2:
        st.metric("📦 Total Orders", total_orders)
    with col3:
        st.metric("👥 Total Customers", total_customers)
    with col4:
        st.metric("🔥 Active Customers", active_customers)
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("📈 Conversion Rate", conversion_rate)
    with col6:
        st.metric("📊 Avg Order Value", AOV)
    with col7:
        st.metric("🔄 Return Rate", f"{return_rate:.2%}")
    with col8:
        st.metric("💸 Life Time Value", f"{LTV:,.2f}")
        
    #-------------------------------------------------------Revenue Trend-------------------------------------------------------
    # Compute Sales Trend with Category Information
    def compute_sales_trend(df):
        df = df.copy()
        df["order_month"] = pd.to_datetime(df["order_date"], errors='coerce').dt.to_period("M").astype(str)
        df["category"] = df["category"].fillna("Miscellaneous")
        trend = df.groupby(["order_month", "category"])["price"].sum().reset_index()
        trend.rename(columns={"order_month": "order_date"}, inplace=True)
        return trend
    
    sales_trend = compute_sales_trend(orders_df)
    
    # Area chart by category
    fig_sales = px.area(
        sales_trend,
        x="order_date", y="price", color="category",
        title="📈 Revenue Breakdown Over Time",
        markers=True, line_shape="spline", template="plotly_dark",
        labels={"price": "Revenue", "order_date": "Time", "category": "Category"}
    )
    
    # fig_sales.update_layout(title_x=0.2)
    
    # Add total revenue trend line
    total_revenue_trend = sales_trend.groupby("order_date")["price"].sum().reset_index()
    total_revenue_trend["numeric_date"] = pd.to_datetime(total_revenue_trend["order_date"]).view('int64') // 10**9
    
    model = LinearRegression()
    X = total_revenue_trend[["numeric_date"]]
    y = total_revenue_trend["price"]
    model.fit(X, y)
    total_revenue_trend["trend"] = model.predict(X)
    
    fig_sales.add_trace(
        go.Scatter(
            x=total_revenue_trend["order_date"],
            y=total_revenue_trend["trend"],
            mode="lines",
            name="Total Revenue Trend",
            line=dict(color="black", width=3, dash="dash")
        
        )
    )
    fig_sales.update_layout(
        legend=dict(
            orientation="v",
            y=1.3,
            x=0.9,
            xanchor="center"
        )
        , title_font=dict(size=20)
    )
    
    #-------------------------------------------------------Category Pie-------------------------------------------------------
    cat_subcat_rev = orders_df.groupby(["category", "subcategory"]).agg({"price": "sum"}).reset_index()
    fig_cat_subcat = px.sunburst(
        cat_subcat_rev,
        path=["category", "subcategory"],
        values="price",
        title="🍩 Revenue Breakdown",
        template="plotly_white",
        color="category",
    #     color_discrete_map={"Accessories": "#D8BFD8"},
        labels={"price": "Revenue"},
        hover_data={"price": True},
        branchvalues="total",
        custom_data=["price"]
    )
    fig_cat_subcat.update_layout(
        margin=dict(t=60, l=0, r=0, b=0),
        legend=dict(title="Category", orientation="h", y=1.1, x=0.5, xanchor='center'),
        uniformtext=dict(minsize=10, mode='hide')
        , title_font=dict(size=20)
    )
    
    # Key Metrics
    col1, col2 = st.columns([2,1])
    with col1:
        st.plotly_chart(fig_sales, use_container_width=True)
    with col2:
        st.plotly_chart(fig_cat_subcat, use_container_width=True)
    
    #-------------------------------------------------------Session Trend-------------------------------------------------------
    # Prepare session data
    sessions_df["session_month"] = pd.to_datetime(sessions_df["timestamp"]).dt.to_period("M").astype(str)
    
    # Compute session trends by traffic source
    session_trend = sessions_df.groupby(["session_month", "source"])["session_id"].count().reset_index()
    session_trend.rename(columns={"session_month": "session_date", "session_id": "sessions"}, inplace=True)
    
    # Create area chart
    fig_sessions = px.area(
        session_trend,
        x="session_date", y="sessions", color="source",
        title="🌐 Session Trend by Traffic Source",
        markers=True, line_shape="spline", template="plotly_dark",
        labels={"sessions": "Session Count", "session_date": "Time", "source": "Traffic Source"}
    )
    
    # Add total session trend line
    total_session_trend = session_trend.groupby("session_date")["sessions"].sum().reset_index()
    total_session_trend["numeric_date"] = pd.to_datetime(total_session_trend["session_date"]).view('int64') // 10**9
    
    model = LinearRegression()
    X = total_session_trend[["numeric_date"]]
    y = total_session_trend["sessions"]
    model.fit(X, y)
    total_session_trend["trend"] = model.predict(X)
    
    fig_sessions.add_trace(
        go.Scatter(
            x=total_session_trend["session_date"],
            y=total_session_trend["trend"],
            mode="lines",
            name="Total Session Trend",
            line=dict(color="black", width=3, dash="dash")
        )
    )
    
    fig_sessions.update_layout(
        legend=dict(
            orientation="v",
            y=1.4,
            x=0.9,
            xanchor="center"
        )
        , title_font=dict(size=20)
    )
    #-------------------------------------------------------Funnel Plots-------------------------------------------------------
    sessions_df["converted"] = sessions_df["converted"].astype(bool)
    sessions_df["has_cart"] = sessions_df["page_views"] >= 3
    sessions_df["checkout_started"] = sessions_df["page_views"] >= 6
    
    funnel_counts = {
        "Sessions": len(sessions_df),
        # "Product Views": sessions_df[sessions_df["page_views"] >= 1].shape[0],
        "Add to Cart": sessions_df[sessions_df["has_cart"]].shape[0],
        "Checkout Started": sessions_df[sessions_df["checkout_started"]].shape[0],
        "Order Placed": sessions_df[sessions_df["converted"]].shape[0]
    }
    
    fig_funnel = go.Figure(go.Funnel(
        y=list(funnel_counts.keys()),
        x=list(funnel_counts.values()),
        textinfo="value+percent initial",
        marker={"color": ["#f8cb2e", "#bfbfbf", "#f49c6e", "#5ba8d0", "#4b8bbe"]}
    ))
    
    fig_funnel.update_layout(title="🔻 Funnel Based on Session Data", title_font=dict(size=20))
    
    # Key Metrics
    col1, col2 = st.columns([2,1])
    with col1:
        st.plotly_chart(fig_sessions, use_container_width=True)
    with col2:
        st.plotly_chart(fig_funnel, use_container_width=True)    
#-------------------------------------------------------Customer Segmentation-------------------------------------------------------
# with tab2:
elif section == "🎯 Customer Segmentation":

    # Dashboard Layout
    st.markdown("<h3 style='text-align: center;'> 🎯 Customer Segmentation Analytics </h3>", unsafe_allow_html=True)
    
    seg_table = pd.read_csv("Segmentation Summary.csv")
    
    # Build Plot
    seg_table["text_label"] = (
        seg_table["cluster"].astype(str) +
        "<br>N=" + seg_table["count"].astype(str)
    )
    
    fig = px.scatter(
        seg_table,
        x="mean_tenure",
        y="mean_recency_days",
        size="sum_customer_lifetime_value",
        size_max=80, 
        color="mean_high_value_order_ratio",
        color_continuous_scale='RdBu_r',
        text="text_label",                       # increase bubble size range (default is 20)
        title="🧩 Customer Segmentation Distribution",
        labels={
            "mean_tenure": "Tenure (days)",
            "mean_recency_days": "Recency (days)",
            "mean_high_value_order_ratio": "High-Value Ratio"
        }
    )
    
    positions = np.where(seg_table['mean_recency_days'] < 120, 'top center', 'bottom center')  
    fig.update_traces(textposition=positions,
                      marker=dict(opacity=0.7),  # adjust transparency here,
                      textfont=dict(
                            size=12,
                            color='black',
                            family='Arial'
                        )
                     )
    
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        title_x=00,  # center title
        title_font=dict(size=20),
        margin=dict(l=60, r=60, t=100, b=60),
        font=dict(family="Arial", size=20),
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    
    fig.add_annotation(
        text="Bubble size = Total Monetary Value, Color = High-Value Order Ratio",
        xref="paper", yref="paper",
        x=0.55, y=1.1,  # above title
        showarrow=False,
        font=dict(size=14),
        xanchor='center'
    )
    
    # Streamlit App
    st.plotly_chart(fig, use_container_width=True)
    
    # GPT model
    import os
    import openai
    import holidays
    from datetime import date, timedelta
    from openai import OpenAI
    
    def get_seasonal_campaign_prompt(seg_table_df, api_key):
        # Detect today's date & season
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
    
        # Detect upcoming US holidays (next 45 days)
        us_holidays = holidays.UnitedStates()
        upcoming = {
            d: name for d, name in us_holidays.items()
            if today <= d <= today + timedelta(days=45)
        }
        holiday_context = "\n".join(
            f"- {name} on {d.strftime('%B %d')}" for d, name in sorted(upcoming.items())
        ) or "No major holidays in the next 45 days."
    
        # Retail seasonal occasions
        seasonal_occasion_context = {
            "Spring": "🌸 Spring cleaning, Mother's Day, Graduation season",
            "Summer": "☀️ Summer kick-off, Wedding season, Pride Month",
            "Fall": "🍂 Back to School, Fall layering, Halloween preview",
            "Winter": "❄️ Holiday gifting, New Year reset, Winter sales"
        }
    
        # Format cluster summary table
        seg_table_text = seg_table_df.to_string(index=False)
    
    
        # Build prompt
        prompt = f"""
    You are a customer insights storyteller and luxury marketing strategist for a high-end retail brand.
    
    📅 Today is {today.strftime('%B %d, %Y')}.
    
    You’ve received clustering results based on customer behavior, including:
    - recency (days since they recently purchased)
    - high-value order ratio (how often they purchase premium items)
    - tenure (how long they’ve been a customer)
    - customer lifetime value (CLV)
    - consistency in purchase value (std_order_value)
    
    Each cluster also includes:
    - their favorite **subcategory** of products (e.g. Cardigan)
    - a list of their **top 5 favorite product IDs** from recent shopping behavior
    
    Your first task is to write a cohesive and inspiring **persona summary** for each cluster, which should include:
    
    1. ✨ A short, clear **persona description**: What are they like? What’s their vibe? 
    2. 💌 A personalized **email campaign suggestion** — feel free to include subject lines, tone, themes, or offer style
    3. 📱 A complementary **social media campaign idea** — reels, challenges, influencer content, or lookbook themes
    4. 🧣 List their fav subcategory and product ids from the segmentation summary table for reference.
    
    Your second task is to:
    🌸 Tie your suggestions to the **current season ({season})** and any **upcoming holidays or retail moments**
    1 row for 1 season/ holiday events
    
    For the 2 tasks, the requirements are:
    🎯 The tone should be strategic, and highly summarizable
    📝 Use Markdown formatting and emojis. Write for a brand team preparing a campaign launch — it should feel inspiring, polished, and presentation-ready.
    
    ---
    
    📊 Here is the cluster summary table (one row per cluster):
    {seg_table_df}
    
    🛍️ Seasonal Retail Moments:
    {seasonal_occasion_context[season]}
    
    📅 Upcoming U.S. Holidays:
    {holiday_context}
    """
        return prompt
    
    def generate_campaign_suggestions(seg_table_df, api_key):
        # Build OpenAI client (new v1.0+ syntax)
        client = OpenAI(api_key=api_key)
    
        # Build prompt with seasonal logic
        prompt = get_seasonal_campaign_prompt(seg_table_df, api_key)
    
        # Send to GPT
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a retail marketing strategist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
    
        return response.choices[0].message.content
        

    
    
    with st.expander("🤖 GPT Segment Interpreter"):
        st.markdown("Ask GPT to interpret your customer segments:")
        if st.button("Generate Interpretation"):
            st.info("Generating GPT summary...")
            gpt_response = generate_campaign_suggestions(seg_table,None)
            st.markdown(gpt_response)


