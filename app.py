import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objs as go
import catboost
import gdown
from sklearn.metrics import mean_squared_log_error, r2_score

url = 'https://drive.google.com/uc?id=1uDpF9_kJCD60aqZzXFoukxurbNhjLdAS'
output = 'models_per_family.pkl'
gdown.download(url, output, quiet=True)
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
        for lag in [16, 30, 60]:  # Lặp qua các giá trị lag
            new_df[f"SMA{i}_sales_lag{lag}"] = (
                new_df.groupby(["store_nbr", "family"])["sales"]
                .transform(lambda x: x.rolling(i).mean().shift(lag))
            )

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
    # st.experimental_rerun() 
    # st.title("TEST RESET")
    # st.write("Nếu bạn thấy này nghĩa là đã reset thành công")
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

        # Set sales for the last 15 days to NaN
        last_date = df['date'].max()
        start_date = df["date"].min() 
        last_15_days = pd.date_range(end=last_date, periods=15)
        # df.loc[df['date'].isin(last_15_days), 'sales'] = None  # Set to NaN

        

        # Merge input data with additional datasets
        sales_merged = df.merge(
            df_store, on="store_nbr", how="left",
        ).merge(
            local, on=["date", "city"], how="outer",
        ).merge(
            regional, on=["date", "state"], how="outer",
        ).merge(
            national, on="date", how="outer",
        ).merge(
            events, on="date", how="outer",
        )
        
        
     
         # Create dummy variables
        column_4 = ['holiday_local', 'holiday_regional', 'holiday_national', 'events']
        sales_merged, new_columns = one_hot_encode(df=sales_merged, columns=column_4, nan_dummie=False, dropfirst=False)
        sales_merged = sales_merged[(sales_merged['date'] <= last_date) & (sales_merged['date'] >= start_date)]
        #sales_merged = sales_merged.sort_values(by=["id"])
       
        
        # Create trend variable
        sales_merged['trend'] = (sales_merged['date'] - pd.Timestamp('2013-01-01')).dt.days + 1
        
        # Create day_name feature
        sales_merged['day_name'] = sales_merged['date'].dt.strftime('%A')
        
        # Create lag features
        sales_merged = create_lags(sales_merged)
        
        # Create rolling mean features
        sales_merged = create_rolling_mean(sales_merged)
        
        
        
        # Create dummy variables
        # column_4 = ['holiday_local', 'holiday_regional', 'holiday_national', 'events']
        # sales_merged, new_columns = one_hot_encode(df=sales_merged, columns=column_4, nan_dummie=False, dropfirst=False)
        
        cols = sales_merged.columns.tolist()
        cols_reordered = cols[:10] + cols[30:] + cols[10:30]
        # Sắp xếp lại DataFrame
        sales_merged = sales_merged[cols_reordered]
        #sales_merged = sales_merged.sort_values(by=["id"])
        # st.write(sales_merged) 
        
        # Move specified columns to category type
        cat_features  = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster', 'day_name']
        sales_merged['store_nbr'] = sales_merged['store_nbr'].astype(int)
        sales_merged['cluster'] = sales_merged['cluster'].astype(int)
        sales_merged[cat_features] = sales_merged[cat_features].astype(str)
        
        # Select all features except 'id', 'date', and 'sales' for model input (last 15 days)
        model_input = sales_merged[sales_merged['date'].isin(last_15_days)].drop(columns=['id', 'date', 'sales'])

       
        # for col in category_columns:
        #     model_input[col] = model_input[col].astype('category')
       # Hoặc .astype(int)    
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
               
                # st.write(input_per_family.shape) 
                # st.write(input_per_family)
                # st.title("Xem mô hình")
                # st.write(catboost.__version__)  # Kiểm tra phiên bản CatBoost hiện tại
                # st.write(models_per_family[i].feature_names_)  # Xem danh sách tất cả các biến đầu vào
                # st.write(models_per_family[i].get_params())  # Xem tất cả tham số của mô hình
                # st.title("Xem biến cate")
                # st.write("Training categorical features:", models_per_family[i].get_all_params().get('cat_features', []))
                # st.write("Input data categorical features:", input_per_family.select_dtypes(include=['category']).columns.tolist())
                # st.write(input_per_family.dtypes)
                # try:
                #     prediction = models_per_family[i].predict(input_per_family, task_type="CPU")
                  
                predict_value = models_per_family[i].predict(input_per_family)
                # Set predicted values less than 0 to 0
                predict_value[predict_value < 0] = 0
                # Apply the exponential function to un-log the values (adding 1 to avoid log(0))
                predict_value = np.exp(predict_value )  # Use np.exp for natural logarithm
                predictions.append(predict_value)
        
        # Concatenate predictions into child_df
        for i, value in enumerate(family_names):
            input_per_family = model_input[model_input['family'] == value]
            if not input_per_family.empty:
                child_df.loc[child_df['family'] == value, 'predicted_sales'] = predictions[i]

         # Merge giá trị thực tế vào child_df để so sánh
        child_df = child_df.merge(
            sales_merged[['date', 'store_nbr', 'family', 'sales']],  # Cột thực tế
            on=['date', 'store_nbr', 'family'],
            how='left'
        )
        

               # Sau khi có child_df với predicted_sales
        
        # 1. Đổi tên cột thực tế
        child_df = child_df.rename(columns={'sales': 'actual_sales'})
        
        # 2. Xử lý dữ liệu
        # child_df = child_df[
        #     (child_df['actual_sales'] > 0) & 
        #     (child_df['predicted_sales'] > 0)
        # ]
        # child_df = child_df.dropna(subset=['actual_sales', 'predicted_sales'])
        
        # 3. Tính RMSLE
        if not child_df.empty:
            rmsle = np.sqrt(mean_squared_log_error(
                child_df['actual_sales'],
                child_df['predicted_sales']
            ))
            r2 = r2_score(
            child_df['actual_sales'],
            child_df['predicted_sales']
            )
            st.write("## Đánh giá hiệu suất")
            st.write("## Đánh giá hiệu suất")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("RMSLE", f"{rmsle:.4f}")
            
            with col2:
                st.metric("R² Score", f"{r2:.4f}")
            
            with col3:
                st.metric("Số mẫu hợp lệ", len(child_df))
        else:
            st.warning("Không có dữ liệu hợp lệ để tính RMSLE")
        
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
