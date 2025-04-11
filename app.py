import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objs as go
import catboost
import gdown
from sklearn.metrics import mean_squared_log_error, r2_score
import plotly.express as px

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
    st.title("Hệ thống dự báo doanh thu")
    st.markdown("""
    ### 📄 Hướng dẫn định dạng tập tin đầu vào (.csv)
    
    Tập tin dữ liệu đầu vào cần có cấu trúc giống với tập huấn luyện, bao gồm các cột sau:
    
    - `id`: Mã định danh duy nhất cho mỗi dòng dữ liệu
    - `date`: Ngày giao dịch (định dạng: YYYY-MM-DD)
    - `store_nbr`: Số thứ tự cửa hàng
    - `family`: Nhóm sản phẩm
    - `sales`: Doanh thu (có thể để trống hoặc không)
    - `onpromotion`: Số lượng sản phẩm đang khuyến mãi
    
    📌 **Lưu ý**:
    - Tập tin phải chứa ít nhất **15 ngày dữ liệu gần nhất** cho mỗi cửa hàng và nhóm sản phẩm.
    - Nếu cột `sales` cho 15 ngày cuối **để trống (NaN)** → hệ thống sẽ dự báo doanh thu.
    - Nếu đã có giá trị thực tế → hệ thống sẽ dự báo doanh thu và vẽ thêm biểu đồ so sánh và tính RMSLE, R².
    
    """)
    # Upload file
    uploaded_file = st.file_uploader("Chọn tệp tin CSV", type=["csv"])
    
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
        sales_merged = sales_merged.sort_values(by=["id"])
       
        
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
        sales_merged = sales_merged.sort_values(by=["id"])
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
                predict_value = models_per_family[i].predict(input_per_family)
                # Set predicted values less than 0 to 0
                predict_value[predict_value < 0] = 0
                # Apply the exponential function to un-log the values (adding 1 to avoid log(0))
                predict_value = np.expm1(predict_value)    # Use np.exp for natural logarithm
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
        
        # 2. Kiểm tra dữ liệu thực tế có đủ để đánh giá không
        valid_eval_df = child_df.dropna(subset=['actual_sales', 'predicted_sales'])
        
        # 3. Tính RMSLE và R² nếu có dữ liệu thực tế
        if not valid_eval_df.empty:
            rmsle = np.sqrt(mean_squared_log_error(
                valid_eval_df['actual_sales'],
                valid_eval_df['predicted_sales']
            ))
            r2 = r2_score(
                valid_eval_df['actual_sales'],
                valid_eval_df['predicted_sales']
            )
            st.write("## Đánh giá hiệu suất")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMSLE", f"{rmsle:.4f}")
            with col2:
                st.metric("R² Score", f"{r2:.4f}")
            with col3:
                st.metric("Số mẫu hợp lệ", len(valid_eval_df))
        else:
            st.warning("Không có dữ liệu thực tế để đánh giá hiệu suất (RMSLE, R²).")
        
        # -- Vẽ biểu đồ theo store và family đã chọn --
        selected_store = st.selectbox("Chọn Cửa hàng", options=sales_merged['store_nbr'].unique())
        selected_family = st.selectbox("Chọn Măt hàng", options=sales_merged['family'].unique())
        
        if st.button("Vẽ biểu đồ dự báo"):
            filtered_predictions = child_df[(child_df['store_nbr'] == selected_store) & (child_df['family'] == selected_family)]
        
            # Chỉ chọn các cột có dữ liệu không toàn NaN
            columns_to_plot = ['predicted_sales']
            if filtered_predictions['actual_sales'].notna().any():
                columns_to_plot.append('actual_sales')
        
            # Reshape để vẽ
            df_melted = filtered_predictions.melt(
                id_vars=['date'], 
                value_vars=columns_to_plot, 
                var_name='Type', 
                value_name='Revenue'
            )
            # 💡 Đổi tên để hiển thị tiếng Việt trong legend
            df_melted['Type'] = df_melted['Type'].map({
                'predicted_sales': 'Dự báo',
                'actual_sales': 'Thực tế'
            })
            # Vẽ biểu đồ
            fig = px.line(df_melted, x='date', y='Revenue', color='Type',
                          title=f'Doanh thu dự báo cho cửa hàng {selected_store} và mặt hàng {selected_family} trong 15 ngày',
                          labels={'Revenue': 'Doanh thu', 'date': 'Thời gian', 'Type': 'Legend'},
                          template='plotly_white')
        
            min_date = df_melted['date'].min()
            max_date = df_melted['date'].max()
            date_range_padding = (max_date - min_date) * 0.1
        
            fig.update_layout(
                xaxis=dict(
                    range=[min_date - date_range_padding, max_date + date_range_padding]
                )
            )
        
            st.plotly_chart(fig)




        
        # Export button
        if st.button("Xuất tệp tin dự báo ra định dạng CSV"):
            child_df.to_csv('predicted_revenue.csv', index=False)
            st.success("Data exported successfully!")


if __name__ == "__main__":
    main()
