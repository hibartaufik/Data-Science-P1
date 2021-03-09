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
   ```
   #import data
   train = pd.read_csv('/content/data latih.csv')
   test = pd.read_csv('/content/data test.csv')
   ```
3. Cek 5 data teratas
   ```
   train.head()
   ```
   ![image](https://user-images.githubusercontent.com/74480780/110502864-2be77580-812e-11eb-9610-75421c2053f4.png)

### b. Exploratory Data Analysis (EDA)
Menganalisa karakteristik data dengan fungsi head(), info(), describe(), shape, dan beberapa perintah lainnya agar menemukan insight yang dapat berguna dalam pengolahan data dan perancangan model machine learning. Lalu, mencatat segala macam penemuan pada dataset seperti data yang kosong, tidak lengkap, redundant, atau data yang perlu pengolahan lebih lanjut. Hal-hal yang sudah dicatat tersebut akan diolah dan dieksekusi pada tahapan Data Preprocessing.
1. Cek 5 data teratas

   ```
    train.head()
   ```
   ![image](https://user-images.githubusercontent.com/74480780/110502864-2be77580-812e-11eb-9610-75421c2053f4.png)

2. Cek jumlah dan tipe data pada setiap kolom dataset
   ```
   train.info
   ```
   ![image](https://user-images.githubusercontent.com/74480780/110501825-289fba00-812d-11eb-8bae-fd9d61c4bcd5.png)

3. Cek statistic summary dari dataset
   ```
   train.describe(include='all').T
   ```
   ![image](https://user-images.githubusercontent.com/74480780/110504913-386ccd80-8130-11eb-8442-0e9ac9f5813d.png)

4. Cek bentuk dimensi dari dataset
   ```
   train.shape
   ```
   ![image](https://user-images.githubusercontent.com/74480780/110504983-4b7f9d80-8130-11eb-9831-b167e9765443.png)
   
5. Melihat apa ada data yang kosong pada setiap kolom
   ```
   train.isnull().sum() 
   ```
   ![image](https://user-images.githubusercontent.com/74480780/110505283-97324700-8130-11eb-87e8-4b25d4f5becf.png)
   
6. Melihat urutan pasien berdasarkan umur
   ```
   train.sort_values('umur', ascending=False)
   ```
   ![image](https://user-images.githubusercontent.com/74480780/110505485-c779e580-8130-11eb-9c4d-426944e96c13.png)

7. Melihat jumlah pasien berdasarkan gender dengan visualisasi bar plot
   ```
   plt.style.use('ggplot')
   fg, ax = plt.subplots(figsize=(12,6))
   sns.countplot(x=train['jenis_kelamin'])
   plt.title("JUMLAH PASIEN BERDASARKAN GENDER", pad=20, fontsize=20, fontweight='bold')
   plt.xlabel("Gender", fontsize=14)
   plt.ylabel("Jumlah Pasien", fontsize=14)
   plt.show()
   ```
   ![image](https://user-images.githubusercontent.com/74480780/110505687-fabc7480-8130-11eb-8717-4c0baf877307.png)
   
8. Melihat hubungan/korelasi antar feature pada dataset
   ```
   train.corr()
   ```
   ![image](https://user-images.githubusercontent.com/74480780/110505841-1e7fba80-8131-11eb-840f-3d48fc146ab9.png)
   
9. Melihat hubungan/korelasi antar feature dengan visualisasi heatmap
   ```
   plt.style.use('ggplot')
   fg, ax = plt.subplots(figsize=(14,6))
   mask = np.triu(train.corr())
   sns.heatmap(train.corr(), cmap='Reds', mask=mask, annot=True, linewidths=2)
   plt.title('KORELASI TIAP FEATURE', pad=20, fontsize=20, fontweight='bold')
   plt.show()
   ```
   ![image](https://user-images.githubusercontent.com/74480780/110506005-496a0e80-8131-11eb-8d15-2500c8ff8db1.png)
   
10. Melihat jumlah data pada data feature yang akan diprediksi (target/label)
    ```
    train['stroke'].value_counts()
    ```
   ![image](https://user-images.githubusercontent.com/74480780/110506191-74546280-8131-11eb-8c61-cd2828095246.png)









