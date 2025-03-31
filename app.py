import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objs as go
import catboost


def create_lags(df):
    new_df = df.copy()
    keys = ['store_nbr', 'family']
    val = 'sales'
    lags = [16, 21, 30, 45, 60, 90]

    for lag in lags:
        new_df['lag_' + str(lag)] = new_df.groupby(keys)[val].transform(lambda x: x.shift(lag))

    return new_df

def create_rolling_mean(df):
    new_df = df.sort_values(["store_nbr", "family", "date"]).copy()

    for i in [20]: 
        new_df["SMA" + str(i) + "_sales_lag16"] = new_df.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(16).values 
        new_df["SMA" + str(i) + "_sales_lag30"] = new_df.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(30).values 
        new_df["SMA" + str(i) + "_sales_lag60"] = new_df.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(60).values 

    return new_df

def one_hot_encode(df, columns, nan_dummie=True, dropfirst=False): 
    original_columns = list(df.columns) 
    dummie_columns = columns 
    df = pd.get_dummies(df, columns=dummie_columns, dummy_na=nan_dummie, drop_first=dropfirst) 
    df.columns = df.columns.str.replace(" ", "_") 
    new_columns = [c for c in df.columns if c not in original_columns] 
    return df, new_columns

def main():
    st.title("CSV File Reader")
    
    # Upload file
    uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])
    
    if uploaded_file is not None:
        # Đọc file CSV
        df = pd.read_csv(uploaded_file)
        
        # Convert date column to datetime type
        df['date'] = pd.to_datetime(df['date'])  # Ensure the date column is in datetime format
        
        # Hiển thị dataframe
        st.write("### Nội dung của file:")
        st.dataframe(df)
        
        # Hiển thị một số thông tin
        st.write("### Thống kê nhanh:")
        st.write(df.describe())
        
        # Prepare input data for revenue prediction
        # Load additional CSV files
        df_store = pd.read_csv("stores.csv")  # Updated to use stores.csv
        local = pd.read_csv("local.csv")          # Assuming the file is named local.csv
        regional = pd.read_csv("regional.csv")    # Assuming the file is named regional.csv
        national = pd.read_csv("national.csv")    # Assuming the file is named national.csv
        events = pd.read_csv("events.csv")        # Assuming the file is named events.csv
        
        # Convert date columns to datetime type
        local['date'] = pd.to_datetime(local['date'])
        regional['date'] = pd.to_datetime(regional['date'])
        national['date'] = pd.to_datetime(national['date'])
        events['date'] = pd.to_datetime(events['date'])
        
        # Merge input data with additional datasets
        sales_merged = df.merge(
            df_store, on="store_nbr", how="left",
        ).merge(
            local, on=["date", "city"], how="left",
        ).merge(
            regional, on=["date", "state"], how="left",
        ).merge(
            national, on="date", how="left",
        ).merge(
            events, on="date", how="left",
        )
        
        # Set sales for the last 15 days to NaN
        last_date = sales_merged['date'].max()
        last_15_days = pd.date_range(end=last_date, periods=15)
        sales_merged.loc[sales_merged['date'].isin(last_15_days), 'sales'] = None  # Set to NaN
        
        # Create trend variable
        sales_merged['trend'] = (sales_merged['date'] - pd.Timestamp('2013-01-01')).dt.days + 1
        
        # Create day_name feature
        sales_merged['day_name'] = sales_merged['date'].dt.strftime('%A')
        
        # Create lag features
        sales_merged = create_lags(sales_merged)
        
        # Create rolling mean features
        sales_merged = create_rolling_mean(sales_merged)
        
        # Move specified columns to category type
        category_columns = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster', 'day_name']
        for col in category_columns:
            sales_merged[col] = sales_merged[col].astype('category')
        
        # Create dummy variables
        column_4 = ['holiday_local', 'holiday_regional', 'holiday_national', 'events']
        sales_merged, new_columns = one_hot_encode(df=sales_merged, columns=column_4, nan_dummie=False, dropfirst=False)
        
        # Select all features except 'id', 'date', and 'sales' for model input (last 15 days)
        model_input = sales_merged[sales_merged['date'].isin(last_15_days)].drop(columns=['id', 'date', 'sales'])
        
        # Create child DataFrame containing only date, store_nbr, and family for the last 15 days
        child_df = sales_merged[sales_merged['date'].isin(last_15_days)][['date', 'store_nbr', 'family']]
        
        # Load the model
        models_per_family = joblib.load('models_per_family.pkl')
        
        # Prepare predictions
        predictions = []
        family_names = sales_merged['family'].unique()  # Get unique family names
        
        for i, value in enumerate(family_names):
            input_per_family = model_input[model_input['family'] == value]
            if not input_per_family.empty:
                predict_value = models_per_family[i].predict(input_per_family)
                # Set predicted values less than 0 to 0
                predict_value[predict_value < 0] = 0
                # Apply the exponential function to un-log the values (adding 1 to avoid log(0))
                predict_value = np.exp(predict_value + 1)  # Use np.exp for natural logarithm
                predictions.append(predict_value)
        
        # Concatenate predictions into child_df
        for i, value in enumerate(family_names):
            input_per_family = model_input[model_input['family'] == value]
            if not input_per_family.empty:
                child_df.loc[child_df['family'] == value, 'predicted_sales'] = predictions[i]
        
        # Selection interface for store_nbr and family
        selected_store = st.selectbox("Select Store Number", options=sales_merged['store_nbr'].unique())
        selected_family = st.selectbox("Select Family", options=sales_merged['family'].unique())
        
        # Prediction button
        if st.button("Predict Revenue"):
            # Filter predictions based on selected store and family
            filtered_predictions = child_df[(child_df['store_nbr'] == selected_store) & (child_df['family'] == selected_family)]
            
            # Plotting the predictions using Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_predictions['date'],
                y=filtered_predictions['predicted_sales'],
                mode='lines+markers',
                name='Predicted Revenue'
            ))
            fig.update_layout(
                title=f'Predicted Revenue for Store {selected_store} and Family {selected_family} in the Next 15 Days',
                xaxis_title='Date',
                yaxis_title='Predicted Revenue',
                xaxis=dict(tickformat='%Y-%m-%d'),
                template='plotly_white'
            )
            st.plotly_chart(fig)
        
        # Export button
        if st.button("Export to CSV"):
            child_df.to_csv('predicted_revenue.csv', index=False)
            st.success("Data exported successfully!")

        # Display the model input dataframe
        # st.write("### Model Input DataFrame for the Last 15 Days:")
        # st.dataframe(model_input)

        # Display the child dataframe with predictions
        # st.write("### Child DataFrame for the Last 15 Days with Predictions:")
        # st.dataframe(child_df)

        # Display the merged dataframe
        # st.write("### Merged DataFrame for Revenue Prediction:")
        # st.dataframe(sales_merged)

if __name__ == "__main__":
    main()
