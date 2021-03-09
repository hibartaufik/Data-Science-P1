# Hibar_Portfolio
Data science portfolio

## [Project 1: Stroke Prediction](https://github.com/hibartaufik/Stroke-Prediction) [Waroenk Skill Bootcamp Competition on Kaggle]
Overview:
1. Membuat model machine learning yang memprediksi pengidap stroke berdasarkan data yang ada 
2. Data yang disediakan yaitu data train dan data test
3. Data yang digunakan dalam pengolahan memiliki 12 kolom
   - id_pasien
   - jenis_kelamin
   - umur
   - hipertensi
   - penyakit_jantung
   - sudah_menikah
   - jenis_pekerjaan
   - jenis_tempat_tinggal
   - rata2_level_glukosa
   - bmi
   - merokok
   - stroke 
5. Data targetnya ialah data 'stroke'
6. Tahapan membuat model machine learning terbagi ke dalam 6 tahap, yaitu:
   - Data Preparation
   - Exploratory Data Analysis
   - Data Preprocessing
   - Create Machine Learning Model
   - Model Evaluation
   - Predict Test Data
   
   Tahapan di atas merupakan acuan yang digunakan untuk membuat model Machine Learning, tahapan tidak baku, dapat disesuaikan berdasarkan karakteristik data dan studi kasus
7. Project berasal dari Tugas Besar Bootcamp 'Waroenk Skill #3 - Python Data Science 101'.
   - Repository project pada GitHub dapat diakses [disini](https://github.com/hibartaufik/Stroke-Prediction)
   - Repository projct pada Google Colab dapat diakses disini [disini](https://colab.research.google.com/drive/1mZCqeNFj02YfWY0VlEOnVJfC6EEsQqMm?usp=sharing)
   - Official Website Waroenk Skill dapat dilihat [disini](http://waroenkskill.id/)

### a. Data Preparation
1. Import semua library yang akan digunakan
   ```
   #import library yang akan digunakan
   import pandas as pd
   import matplotlib.pyplot as plt

   import seaborn as sns
   import numpy as np

   from sklearn.model_selection import train_test_split
   from sklearn.metrics import classification_report

   from sklearn.metrics import confusion_matrix
   from sklearn.tree import DecisionTreeClassifier
   ```
2. Import dataset yang akan diolah
   ![image](https://user-images.githubusercontent.com/74480780/110493970-b5467a00-8125-11eb-81d3-b0076e2ae11c.png)
3. Cek 5 data teratas
   ![image](https://user-images.githubusercontent.com/74480780/110494395-29811d80-8126-11eb-84ef-769f607d99c0.png)

### b. Exploratory Data Analysis (EDA)
>Mengetahui dengan menganalisa karakteristik data dengan fungsi head(), info(), describe(), shape, dan beberapa perintah lainnya agar menemukan insight yang dapat berguna dalam pengolahan data dan perancangan model machine learning. Lalu, mencatat segala macam penemuan pada dataset seperti data yang kosong, tidak lengkap, redundant, atau data yang perlu pengolahan lebih lanjut. Hal-hal yang sudah dicatat tersebut akan diolah dan dieksekusi pada tahapan Data Preprocessing.
>1. Cek 5 data teratas
    ![image](https://user-images.githubusercontent.com/74480780/110494395-29811d80-8126-11eb-84ef-769f607d99c0.png)
>2. Cek jumlah dan tipe data pada setiap kolom dataset
    ![image](https://user-images.githubusercontent.com/74480780/110495784-6ef21a80-8127-11eb-8f72-7ee669d26766.png)
>3. Cek statistic summary dari dataset
    ![image](https://user-images.githubusercontent.com/74480780/110496285-e9229f00-8127-11eb-951e-eefcfba9ed56.png)
>4. Cek bentuk dimensi dari dataset
    ![image](https://user-images.githubusercontent.com/74480780/110497196-d0ff4f80-8128-11eb-91ad-e5bad8b286d8.png)
>5. Melihat apa ada data yang kosong pada setiap kolom
    ![image](https://user-images.githubusercontent.com/74480780/110497488-11f76400-8129-11eb-99e0-f7bbb2420ba1.png)
>6. Melihat urutan pasien berdasarkan umur
    ![image](https://user-images.githubusercontent.com/74480780/110497820-626ec180-8129-11eb-8a37-99de290db29b.png)
>7. Melihat jumlah pasien berdasarkan gender dengan visualisasi bar plot
    ![image](https://user-images.githubusercontent.com/74480780/110498467-efb21600-8129-11eb-9f6e-2a86eed4c901.png)
    ![image](https://user-images.githubusercontent.com/74480780/110498222-b8dc0000-8129-11eb-91d9-a18f5628a70f.png)
>8. Melihat hubungan/korelasi antar feature pada dataset
    ![image](https://user-images.githubusercontent.com/74480780/110498680-1f611e00-812a-11eb-952e-ed092afb0dd0.png)
>9. Melihat hubungan/korelasi antar feature dengan visualisasi heatmap
    ![image](https://user-images.githubusercontent.com/74480780/110499127-867ed280-812a-11eb-9ad5-045b0a4808d0.png)
    ![image](https://user-images.githubusercontent.com/74480780/110499211-9991a280-812a-11eb-8a09-31a8264ac7bb.png)
>10. Melihat jumlah data pada data feature yang akan diprediksi (target/label)
    ![image](https://user-images.githubusercontent.com/74480780/110499438-ce9df500-812a-11eb-9dc3-358dcc97eb8b.png)









