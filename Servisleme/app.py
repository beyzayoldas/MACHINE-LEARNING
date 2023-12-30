from turtle import color
import streamlit as st
import pandas as pd  
import numpy as np  
import joblib  
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt  
import seaborn as sns




def main():

    st.sidebar.title('Streamlit ile ML Uygulaması')
    selected_page = st.sidebar.selectbox('Sayfa Seçiniz..',["AnaSayfa","Tahmin Yap","İstatislik Görüntüle","Hakkında"])

    if selected_page == "AnaSayfa":
        st.title("Streamlit Uygulamasına Hoşgeldiniz")

        st.markdown(
            """
            Bu proje makine öğrenmesi uygulamalarının web ortamında streamlit
            kullanılarak yayınlanmasına örnek olarak geliştirilmiştir. Bir e-ticaret sitesi üzerinden 696 adet akıllı saat verileri çekilmiş
            ve incelenmiştir. Bu veriler kullanılarak makine öğrenmesi modelleri eğitilmiş ve projeye dahil edilmiştir.

            """
            )
        st.error("Tahmin yapmak, istatislikleri görüntülemek ve proje hakkında daha fazla bilgi edinmek için sol tarafta bulunan menüyü kullanınız.")


    if selected_page == "Tahmin Yap":
        predict()

    if selected_page == "İstatislik Görüntüle":
        eda()

    if selected_page == "Hakkında":
        about()


def about():
    st.title("Geliştirici Bilgileri")
    st.title("Beyza YOLDAŞ")
    st.subheader("GitHub : (http://github.com/beyzayoldas/)")

def eda():
    st.title("İstatislikler")

    data = pd.read_csv('C:\\Users\\user\\OneDrive - Manisa Celal Bayar Üniversitesi\\Masaüstü\\Beyza Yoldaş\\Veri Toplama\\presentation.csv')

    st.header("Bütün Veriler")
    st.dataframe(data)


def predict():

    # Datalar ve modellerin yüklenmesi
    Markalar = load_data_markalar()
    Renkler = load_data_renkler()
    Kordon_renkleri = load_data_Kordon_renkleri()
    


    # Kullanıcı arayüzü ve değer alma
    st.title("Merhaba, Streamlit!")
    selected_marka = marka_index(Markalar, st.selectbox("Marka Seçiniz", Markalar))

    selected_renk = renk_index(Renkler, st.selectbox("Renk Seçiniz", Renkler))

    selected_kordon_renk = kordon_renk_index(Kordon_renkleri, st.selectbox("Kordon Rengi Seçiniz", Kordon_renkleri))

    selected_isletim_tipi = Isletim_tipi(st.radio("İşletim Tipi", ("iOS", "Android", "Android+iOS")))

    selected_sesli_gorusme = Sesli_gorusme(st.radio("Sesli Görüşme Seçiniz", ("Var", "Yok")))
    selected_titresim = Titresim(st.radio("Titreşim Seçiniz", ("Var", "Yok")))
    selected_gps = GPS(st.radio("GPS Seçiniz", ("Var", "Yok")))

    selected_garanti_suresi = Garanti(st.radio("Garanti süresi Seçiniz", ("6 Ay", "1 Yıl", "2 Yıl")))

    selected_kasa_capi = st.number_input("Kasa Çapı Seçiniz", min_value=30, max_value=50)
    st.write("Kasa Çapı: " +str(selected_kasa_capi)+ " mm" )

    selected_batarya_kapasitesi = st.number_input("Batarya Kapasitesi Seçiniz", min_value=50, max_value=1450)
    st.write("Batarya Kapasitesi : " +str(selected_batarya_kapasitesi)+ " mAh" )

    selected_ekran_boyutu = st.number_input("Ekran Boyutu Seçiniz", min_value=1.0, max_value=5.0, format="%.2f")
    st.write("Ekran Boyutu : {} inç".format(selected_ekran_boyutu))


    predict_value = create_prediction_value(selected_marka,selected_kasa_capi, selected_garanti_suresi,selected_batarya_kapasitesi,
                                             selected_ekran_boyutu,selected_sesli_gorusme,selected_renk,selected_titresim,selected_gps,
                                               selected_isletim_tipi,selected_kordon_renk
                                             )
    predict_models = load_models()

    if  st.button("Tahmin Yap"):
        result = predict_models1(predict_models,predict_value)
        if result != None:
            st.success("Tahmin Başarılı")
            st.balloons()
            st.write("Tahmin Edilen Fiyat: " + result + " TL")
        else:
            st.error("Tahmin yaparken hata meydana geldi!")


def load_data_markalar():
    data = pd.read_csv('C:\\Users\\user\\OneDrive - Manisa Celal Bayar Üniversitesi\\Masaüstü\\Beyza Yoldaş\\Veri Toplama\\presentation.csv')

    Markalar = data["Marka"].values
    Markalar = np.unique(Markalar)
    Markalar = pd.DataFrame(data=Markalar, columns=["Marka"])
    return Markalar

def load_data_renkler():
    data = pd.read_csv('C:\\Users\\user\\OneDrive - Manisa Celal Bayar Üniversitesi\\Masaüstü\\Beyza Yoldaş\\Veri Toplama\\presentation.csv')

    Renkler = data["Renk"].values
    Renkler = np.unique(Renkler)
    Renkler = pd.DataFrame(data=Renkler, columns=["Renk"])
    return Renkler

def load_data_Kordon_renkleri():
    data = pd.read_csv('C:\\Users\\user\\OneDrive - Manisa Celal Bayar Üniversitesi\\Masaüstü\\Beyza Yoldaş\\Veri Toplama\\presentation.csv')

    Kordon_renkleri = data["Kordon_renk"].values
    Kordon_renkleri = np.unique(Kordon_renkleri)
    Kordon_renkleri = pd.DataFrame(data=Kordon_renkleri, columns=["Kordon_renk"])
    return Kordon_renkleri



def load_models():

    best_model = joblib.load("C:\\Users\\user\\OneDrive - Manisa Celal Bayar Üniversitesi\\Masaüstü\\Beyza Yoldaş\\Algoritmalar\\akıllısaat_random_forest_model.pkl")
    return best_model

def marka_index(Markalar,marka):
    index = int(Markalar[Markalar["Marka"] == marka].index.values)
    return index

def renk_index(Renkler,renk):
    index = int(Renkler[Renkler["Renk"] == renk].index.values)
    return index

def kordon_renk_index(Kordon_renkleri,kordon_renk):
    index = int(Kordon_renkleri[Kordon_renkleri["Kordon_renk"] == kordon_renk].index.values)
    return index

def Isletim_tipi(Isletim_tipi):
    if Isletim_tipi == "iOS":
        return 1
    else:
        return 0

def Sesli_gorusme(Sesli_gorusme):
    if Sesli_gorusme =="Yok":
        return 0
    else:
        return 1
    
def GPS(GPS):
    if GPS =="Yok":
        return 0
    else:
        return 1

def Titresim(Titresim):
    if Titresim =="Yok":
        return 0
    else:
        return 1

def Garanti(garanti):
    if garanti == "6 Ay":
        return 0 
    if garanti == "1 Yıl":
        return 1
    if garanti == "2 Yıl":
        return 2
    else:
        return 0

def predict_models1(model,res):
    result = str(int(model.predict(res))).strip('[]')
    return result

def create_prediction_value(Marka,
                            Kasa_capi,
                            Garanti_suresi,
                            Batarya_kapasitesi,
                            Ekran_boyutu,
                            Sesli_gorusme,
                            Renk,
                            Titresim,
                            GPS,
                            Isletim_tipi,
                            Kordon_renk):
    res = pd.DataFrame(data = 
            {'Marka':[Marka],
             'Kasa_capi':[Kasa_capi],
             'Garanti_suresi':[Garanti_suresi],
             'Batarya_kapasitesi':[Batarya_kapasitesi],
             'Ekran_boyutu':[Ekran_boyutu],
             'Sesli_gorusme':[Sesli_gorusme],
             'Renk':[Renk],
             'Titresim':[Titresim],
             'GPS':[GPS],
             'Isletim_tipi':[Isletim_tipi],
             'Kordon_renk':[Kordon_renk]})
    return res


if __name__ == "__main__":
    main()