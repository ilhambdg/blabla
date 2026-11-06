import pandas as pd
import numpy as np
import streamlit as st

def baca_data(file):
    df = pd.read_csv(file)
    return df

def buat_matrix(df):
    y = df["harga"] 
    x = df[['luas', 'kamar', 'usia']]
    return np.array(x), np.array(y)

def hitung_norm_equation(x, y):
    XtX = x.T @ x # perkalian matriks
    Xty = x.T @ y

    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        raise ValueError("woi gaboleh") 

    w = np.linalg.solve(XtX, Xty)
    return np.round(w, 6)

def hitung_evaluasi(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    return rmse, r2

def tampil_hasil(x, w, y):
    y_pred = np.dot(x, w)
    y_true = y
    rmse, r2 = hitung_evaluasi(y_true, y_pred)
    XtX = x.T @ x 
    Xty = x.T @ y

    return y_pred, y_true, rmse, XtX, Xty,r2

def predict(x_input, w):
    y_pred = np.dot(x_input, w)
    return y_pred


st.title('praktikum aljabar matriks 26A')

file = st.file_uploader("input data anda dalam bentuk csv")

if file :
    df = baca_data(file)
    st.dataframe(df)
    arr_x, arr_y = buat_matrix(df)
    w= hitung_norm_equation(arr_x, arr_y)

    y_pred, y_true, rmse, XtX, Xty, r2 = tampil_hasil(arr_x, w, arr_y)
    df_y = pd.DataFrame(
        {"y_asli" : y_true, 
         "y_prediksi" : y_pred}
        )

    st.write(f"ini hasil x {arr_x}, shape {len(arr_x)}, {len(arr_x[0])}")
    st.write(f"ini hasil y {arr_y}, shape {len(arr_y)}, {len([arr_y[0]])}")
    st.write(f"hasil transpose x {arr_x.T}, shape {len(arr_x.T)}, {len(arr_x.T[0])}")

    st.write(f"hasil w {w}")

    st.write(f"hasil persamaan anda {w[0]}X {w[1]}X {w[2]}X")

    col_1,  col_2 = st.columns(2)
    with col_1:
        st.subheader("bagian kiri")
        st.write(f"ini hasil rmse {rmse} hasil xtx {XtX} hasil xtv{Xty}")
        st.write(f"hasil r2 {r2}")
    with col_2:
        st.subheader("bagian kanan")
        st.dataframe(df_y)

    luas = st.number_input("Masukkan luas Anda:", step=1)
    kamar = st.number_input("Masukkan kamar Anda:", step=1)    
    usia = st.number_input("Masukkan usia Anda:", step=1)
    
    x_input = np.array([luas, kamar, usia])

    if st.button('prediksi'):
        hasil = predict(x_input, w)
        st.write(f"harga rumah anda {hasil}")


# df = pd.read_csv('data/data_rumah.csv')


# print(df.head(5))
# df.info()
