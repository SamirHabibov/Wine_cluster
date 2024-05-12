# Lazım olan kitabxanalar
from sklearn.base import BaseEstimator, TransformerMixin
from PIL import Image

import plotly.express as px
import streamlit as st
import seaborn as sns
import pandas as pd
import warnings
import pickle
import time

# Potensial xəbərdarlıqların filterlənməsi
warnings.filterwarnings(action = 'ignore')

# Datasetin yüklənməsi
df = pd.read_csv(filepath_or_buffer = 'wine-clustering.csv')

# Sütun adlarının kiçildilməsi və potensial boşluqların silinməsi
df.columns = df.columns.str.strip().str.lower()


# Modelin yüklənməsi
with open(file = 'wine_cluster_model.pickle', mode = 'rb') as pickled_model:
    model = pickle.load(file = pickled_model)

# Şəkilin yüklənməsi
wine_image = Image.open(fp = 'wine.jpeg')
    
# Əsas səhifənin yaradılması
interface = st.container()

# Əsas səhifəyə elementlərin daxil edilməsi
with interface:
    # Səhifənin adının göstərilməsi (səhifə adı --> Wine Quality Clustering)
    st.title(body = 'Wine Quality Clustering')
    
    # Şəkilin göstərilməsi
    st.image(image = wine_image)
    
    # Başlığın göstərilməsi (başlıq adı --> Project Description)
    st.header(body = 'Project Description')
    
    # Proyekt haqqında informasiyanın verilməsi
    st.markdown(body = f"""This is a machine learning project in which wines are clustered based on their quality. 
    KMeans algoritm was used to build the model with **{df.shape[1]}** features. Principal Component Analysis was 
    used to reduce dimensionality whereas the number of clusters was identified using Elbow method.""")
    
    # Kiçik başlığın göstərilməsi (kiçik başlıq adı --> Input Features)
    st.subheader(body = 'Input Features')
    
    # Düz xəttin çəkilməsi
    st.markdown(body = '***')
    
    # Asılı olmayan dəyişənlərin yaradılması (Bütün asılı olmayan dəyişənləri st.slider() metodu ilə yarat)
    alcohol = st.slider(label = 'alcohol', min_value = 11, max_value = 15, value = int(df.alcohol.mean()))
    malic_acid = st.slider(label = 'malic_acid', min_value = 0, max_value = 6, value = int(df.malic_acid.mean()))
    ash = st.slider(label = 'ash', min_value = 1, max_value = 4, value = int(df.ash.mean()))
    ash_alcanity = st.slider(label = 'ash_alcanity', min_value = 10, max_value = 30, value = int(df.ash_alcanity.mean()))
    magnesium = st.slider(label = 'magnesium', min_value = 70, max_value = 162, value = int(df.magnesium.mean()))
    total_phenols = st.slider(label = 'total_phenols', min_value = 0, max_value = 4, value = int(df.total_phenols.mean()))
    flavanoids = st.slider(label = 'flavanoids', min_value = 0, max_value = 6, value = int(df.flavanoids.mean()))
    nonflavanoid_phenols = st.slider(label = 'nonflavanoid_phenols', min_value = 0, max_value = 1, value = int(df.nonflavanoid_phenols.mean()))
    proanthocyanins = st.slider(label = 'proanthocyanins', min_value = 0, max_value = 4, value = int(df.proanthocyanins.mean()))
    color_intensity = st.slider(label = 'color_intensity', min_value = 1, max_value = 13, value = int(df.color_intensity.mean()))
    hue = st.slider(label = 'hue', min_value = 0, max_value = 2, value = int(df.hue.mean()))
    od280 = st.slider(label = 'od280', min_value = 1, max_value = 4, value = int(df.od280.mean()))
    proline = st.slider(label = 'proline', min_value = 278, max_value = 1680, value = int(df.proline.mean()))
    
    # Düz xəttin çəkilməsi
    st.markdown(body = '***')
    
    # Kiçik başlığın göstərilməsi (kiçik başlıq adı --> Making Predictions)
    st.subheader(body = 'Making Predictions')
    
    # Lügət data strukturunun yaradılması
    data_dictionary = {'alcohol':alcohol,
                       'malic_acid':malic_acid,
                       'ash':ash,
                       'ash_alcanity':ash_alcanity,
                       'magnesium':magnesium,
                       'total_phenols':total_phenols,
                       'flavanoids':flavanoids,
                       'nonflavanoid_phenols':nonflavanoid_phenols,
                       'proanthocyanins':proanthocyanins,
                       'color_intensity':color_intensity,
                       'hue':hue,
                       'od280':od280,
                       'proline':proline }
    
    # Lügət data strukturunun DataFrame data strukturuna çevirilməsi
    input_features =  pd.DataFrame(data = data_dictionary, index = [0])
    
    
    # Proqnoz adlarının yaradılması
    cluster_labels = {0:'first', 1:'second', 2:'third'}
    
    # Predict adında düymənin yaradılması
    if st.button('Predict'):
        # Döngünün yaradılması
        with st.spinner(text = 'Sending input features to model...'): 
            # İki saniyəlik pauzanın yaradılması
            time.sleep(2)
            
        # Klasterin model tərəfindən proqnozlaşdırılması
        predicted_cluster = model.predict(X = input_features)[0]
        # Klasterin adının əldə olunması
        cluster_label = cluster_labels.get(predicted_cluster)
        
        # Proqnozun verilməsi ilə bağlı mesajın göstərilməsi
        st.success('Prediction is ready')
        
        # Bir saniyəlik pauzanın yaradılması
        time.sleep(1)
        
        # Proqnozun istifadəçiyə göstərilməsi
        st.markdown(f'Model output: Wine belongs to the **{cluster_label}** cluster')
