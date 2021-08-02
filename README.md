# My Portfolio
Data science portfolio
## Project 3: New Cases Prediction [COVID-19 Indonesia Dataset (Machine Learning)]
Overview:
1. Membuat model machine learning menggunakan metode decision tree berdasarkan dataset yang berisi data COVID-19 di Indonesia
2. Analysis dan Visualisasi yang lebih lengkap dapat dilihat di Project 2 dengan dataset yang sama, project ini lebih berfokus pada pemodelan machine learning
3. Dataset berasal dari [kaggle.com](https://www.kaggle.com/) dengan nama COVID-19 Indonesia Dataset yang disusun oleh Hendratno yang mengambil data dari beberapa sumber yaitu [situs resmi pemerintah SATGAS COVID-19](https://covid19.go.id/), [Badan Pusat Statistik](https://www.bps.go.id/), dan [Hub InaCOVID-19](https://bnpb-inacovid19.hub.arcgis.com/)
4. Dataset disusun berdasarkan time series atau sususan waktu, tingkat nasional, dan tingkat provinsi, juga beserta data demografi dari lokasi/daerah tersebut
5. Dataset memiliki kolom
   - **'Date'** (Tanggal dilaporkan)
   - **'Location ISO Code'** (Kode lokasi berdasarkan standar ISO)
   - **'Location'** (Nama lokasi)
   - **'New Cases'** (Kasus positif harian)
   - **'New Deaths'** (Kasus kematian harian)
   - **'New Daily Recovered'** (Kasus kesembuhan harian)
   - **'New Active Cases'** (Kasus aktif harian)
   - **'Total Cases'** (Jumlah akumulatif kasus positif sampai waktu terkait)
   - **'Total Deaths'** (Jumlah akumulatif kasus kematian sampai waktu terkait)
   - **'Total Recovered'** (Jumlah akumulatif kasus kesembuhan sampai waktu terkait)
   - **'Total Active Cases'** (Jumlah akumulatif kasus aktif sampai waktu terkait)
   - **'Location Level'** (Tingkat lokasi regional atau nasional)
   - **'City or Regency'** (Nama kota atau wilayah)
   - **'Province'** (Nama provinsi lokasi)
   - **'Country'** (Nama negara lokasi)
   - **'Island'** (Nama pulau utama lokasi)
   - **'Time Zone'** (Zona waktu lokasi)
   - **'Special Status'** (Status istimewa lokasi)
   - **'Total Regencies'** (Jumlah kabupaten dalam lokasi terkait)
   - **'Total Cities'** (Jumlah kota dalam lokasi terkait)
   - **'Total Districts'** (Jumlah kecamatan dalam lokasi terkait)
   - **'Total Urban Village'** (Jumlah pedesaan dalam lokasi terkait)
   - **'Total Rural Village'** (Jumlah perkampungan dalam lokasi terkait)
   - **'Area (km2)'** (Area lokasi dalam kilometer persegi)
   - **'Population'** (Jumlah populasi dalam lokasi terkait)
   - **'Population Density'** (Kepadatan penduduk dalam lokasi terkait, rumus = Population / Area)
   - **'Longitude'** (Garis bujur lokasi)
   - **'Latitude'** (Garis lintang lokasi)
   - **'New Cases per Million'** (Rumus = (New Cases / Population) x 1.000.000)
   - **'Total Cases per Million'** (Rumus = (Total Cases / Population) x 1.000.000)
   - **'Total Deaths per Million'** (Rumus = (Total Deaths / Population) x 1.000.000)
   - **'Case Fatality Rate'** (Rumus = (Total Deaths / Total Cases) x 100)
   - **'Case Recovered Rate'** (Rumus = (Total Recovered / Total Cases) x 100)
   - **'Growth Factor of New Cases'** (Kurang dari 1 artinya menurun, 1 artinya tidak ada perubahan, lebih dari 1 artinya meningkat, rumus = Today New Cases / Yesterday New Cases)
   - **'Growth Factor of New Deaths'** (Kurang dari 1 artinya menurun, 1 artinya tidak ada perubahan, lebih dari 1 artinya meningkat, rumus = Today New Deaths / Yesterday New Deaths)
6. Tahapan membuat model machine learning terbagi ke dalam 6 tahap, yaitu:
   - Data Preparation
   - Exploratory Data Analysis
   - Data Preprocessing
   - Modeling
   - Model Evaluation
   - Predict Test Data
   
   Tahapan di atas merupakan acuan yang digunakan untuk membuat model Machine Learning, tahapan tidak baku, dapat disesuaikan berdasarkan karakteristik data dan studi kasus
7. Project menggunakan dataset berasal [kaggle](https://www.kaggle.com/), yang disusun oleh Hendratno.
   - Repository project Github dapat diakses [disini](https://github.com/hibartaufik/New-Cases-Prediction)
   - Dataset dapat diakses [disini](https://www.kaggle.com/hendratno/covid19-indonesia)

### 1. Data Preparation
#### 1.1 Import Libraries
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
```
#### 1.2 Import Dataset
Import dataset yang akan digunakan, lalu tampilkan dataset untuk mengecek apakah import berhasil atau tidak
```
df = pd.read_csv('./dataset/covid_19_indonesia_time_series_all.csv')
df = df.set_index('Location')
```
```
df.head()
```
![image](https://user-images.githubusercontent.com/74480780/126043463-4123b1dc-4d68-40a9-a012-cf87de2ac7e8.png)

### 2. Exploratory Data Analysis (EDA)
Menganalisa karakteristik data dengan perintah `info()`, `describe()`, `shape`, dan beberapa perintah lainnya agar menemukan insight yang dapat berguna dalam pengolahan data dan perancangan model machine learning. Lalu, mencatat segala macam penemuan pada dataset seperti data yang kosong, tidak lengkap, redundant, atau data yang perlu pengolahan lebih lanjut. Hal-hal yang sudah dicatat tersebut akan diolah dan dieksekusi pada tahapan Data Preprocessing.
```
# Melihat jumlah tipe data tiap kolom dalam dataset
df.info()
```
![image](https://user-images.githubusercontent.com/74480780/126043555-b7b82782-3302-4832-a8e0-afd50c7ba05a.png)
![image](https://user-images.githubusercontent.com/74480780/126043559-ae17dde2-b79c-4a48-bf5e-b6700e2d9aab.png)
Dataset memiliki 37 kolom, terdiri dari 12 kolom dengan tipe data object/string, 12 kolom dengan tipe data integer, dan 13 kolom dengan tipe data float.
```
# Melihat summary statistics
df.describe()
```
![image](https://user-images.githubusercontent.com/74480780/126043586-20ed44c0-ec66-47d4-bd93-c74b16854ef6.png)
```
# Melihat dimensi (baris, kolom) dataset
df.shape
```
![image](https://user-images.githubusercontent.com/74480780/126043594-8d9169d1-2982-417f-ac0b-0f16a1759ff7.png)

#### 2.1 Menentukan Kolom yang Akan di-Imputasi
Melihat data null tiap kolom dalam bentuk presentase beserta tipe datanya
```
columns = list(df.columns)

for col in columns:
  print(f"{col}:{round((df[col].isnull().sum() / len(df) * 100), 3)}%\t\t{df[col].dtype}")
```
![image](https://user-images.githubusercontent.com/74480780/126043621-28a2c62a-225a-4c0b-b0bb-5fc14330b415.png)
![image](https://user-images.githubusercontent.com/74480780/126043633-9107a780-ea62-40b2-9852-02853be22e6d.png)
Membuat looping untuk memfilter setiap kolom dalam dataset dan mengelompokannya ke dalam beberapa list untuk menentukan kolom mana saja yang akan diimputasi (kolom yang memiliki jumlah data null < 40%) dan kolom mana yang harus di drop (kolom yang memiliki jumlah data null > 40%) nantinya.

- `cols_null_num` untuk kolom yang memiliki jumlah data null < 40% dan bertipe data numerik (integer & float)
- `cols_null_obj` untuk kolom yang memiliki jumlah data null < 40% dan bertipe data object (string)
- `cols_drop` untuk kolom yang memiliki jumlah data null > 40%

```
cols_null_num = []
cols_null_obj = []
cols_drop = []

for col in columns:
  if ((df[col].isnull().sum() / len(df) * 100) != 0) and ((df[col].isnull().sum() / len(df) * 100) < 40) and (df[col].dtype != 'object'):
    cols_null_num.append(col)
  elif ((df[col].isnull().sum() / len(df) * 100) != 0) and ((df[col].isnull().sum() / len(df) * 100) < 40) and (df[col].dtype == 'object'):
    cols_null_obj.append(col)
  elif ((df[col].isnull().sum() / len(df) * 100) != 0) and ((df[col].isnull().sum() / len(df) * 100) > 40):
    cols_drop.append(col)
```
```
print(f"Kolom numerik: {cols_null_num}")
print(f"Kolom object: {cols_null_obj}")
```
![image](https://user-images.githubusercontent.com/74480780/126043678-b3b09949-6d2d-4ed0-958a-ab90efa31ab4.png)
Kolom yang akan diimputasi:
- Kolom numerik: 'Total Cities', 'Total Urban Villages', 'Total Rural Villages', 'Growth Factor of New Cases', 'Growth Factor of New Deaths'
- Kolom object: 'Province', 'Island', 'Time Zone'

##### 2.1.1 Melihat Distribusi Data Untuk Menetukan Nilai Imputasi Tiap Kolom
- Kolom Numerik
```
for col in cols_null_num:
  sns.displot(df[col].value_counts(), kde=True)
```
![image](https://user-images.githubusercontent.com/74480780/126043744-768d7eca-f1e5-4740-bbb8-71b12772c6fb.png)
![image](https://user-images.githubusercontent.com/74480780/126043773-75b9ca42-6323-42e3-9b24-51fae1176e2b.png)
![image](https://user-images.githubusercontent.com/74480780/126043803-59567879-b64b-45ba-b767-36c5fad6071b.png)
![image](https://user-images.githubusercontent.com/74480780/126043826-292d94c3-e2cb-4eb0-87f1-2baf97acedbf.png)
![image](https://user-images.githubusercontent.com/74480780/126043839-477ab33d-3093-4ea8-b599-b80cd0bf2090.png)
Kita dapat menetukan nilai yang akan mengganti nilai kosong berdasarkan distribusi data tiap kolom. Untuk 'Total Rural Villages' memiliiki data berdistrbusi 'nyaris' normal, nilai modus merupakan pilihan tepat karena data yang bernilai bulat, mengapa tidak mean? karena nilai mean tidak terlalu bisa mewakili distribusi data yang tidak benar-benar normal/merata.

Untuk kolom numerik lain seperti 'Total Cities', 'Total Urban Villages', 'Growth Factor of New Cases', dan 'Growth Factor of New Deaths' memiliki data yang sama sekali tidak merata, sehingga nilai median merupakan nilai yang cocok untuk dapat mengisi data null si setiap kolom-nya.

- Kolom Object
```
for col in cols_null_obj:
  sns.displot(df[col].value_counts(), kde=True)
```
![image](https://user-images.githubusercontent.com/74480780/126043894-8f706f6a-074a-413d-9c4b-b82b4c1cecd8.png)
![image](https://user-images.githubusercontent.com/74480780/126043907-6fb440b6-a003-44b1-9058-d02306bc414a.png)
![image](https://user-images.githubusercontent.com/74480780/126043926-956818ae-9d8a-45ff-8cbb-a5002e816701.png)
Data null pada 'Province', 'Island', dan 'Time Zone' kurang baik jika diisi dengan modus data karena tidak bisa merepresentasikan data kolom terkait seperti provinsi atau pulau saat kasus COVID-19 ditemukan tidak akan dapat direpresentasikan dengan nama provinsi atau pulau yang paling banyak datanya.

Namun solusi drop kolom juga kurang baik karena data null memiliki komposisi yang sedikit. Yang akan dilakukan adalah mengisi nilai null tersebut dengan angka nol, meskipun nilai nol tersebut diisi pada kolom yang bertipe data object (string), namun pada akhirnya kolom tersebut juga akan diubah ke dalam bentuk numerik sehingga pengisian nilai nol tidak akan bermasalah.

#### 2.2 Menentukan Kolom yang Akan di Drop
```
cols_drop
```
![image](https://user-images.githubusercontent.com/74480780/126043955-1c7f85ba-e6f9-478c-9864-626241242958.png)
Kolom yang akan di drop adalah kolom yang memiliki komposisi data null yang banyak (sekitar lebih dari 40% keseluruhan data kolom tersebut). Seluruh data kolom 'City or Regency' adalah null, sedangkan untuk kolom 'Special Status' memiliki 85.569% data null.

### 3. Data Preprocessing

Beberapa hal yang didapatkan dari proses EDA adalah:
- Handling Missing Value:
  
  - Imputasi: 
      
      - 'Total Rural Villages' -> mod
      - 'Total Urban Villages' -> median
      - 'Total Cities' -> median
      - 'Growth Factor of New Cases' -> median 
      - 'Growth Factor of New Deaths' -> median
      - 'Province' -> 0 
      - 'Island' -> 0
      - 'Time Zone' -> 0
  
  - Drop:

    - 'City or Regency'
    - 'Special Status'

- Mengubah Kolom Kategori Menjadi Numerik
#### 3.1 Imputasi
##### 3.1.1 Imputasi Kolom Numerik
```
# Membuat dictionary untuk menampung nilai pengisi
filler_num = {
    'Total Rural Villages': stats.mode(df['Total Rural Villages'])[0][0],
    'Total Urban Villages': df['Total Urban Villages'].median(),
    'Total Cities': df['Total Cities'].median(),
    'Growth Factor of New Cases': df['Growth Factor of New Cases'].median(),
    'Growth Factor of New Deaths': df['Growth Factor of New Deaths'].median()
}
```
```
df.fillna(filler_num, inplace=True)
```
##### 3.1.2 Imputasi Kolom Kategorik
Mengisi data null di kolom 'Province', 'Island', 'Time Zone' dengan nilai nol
```
df[cols_null_obj] = df[cols_null_obj].replace([np.inf, -np.inf], np.nan)
df[cols_null_obj] = df[cols_null_obj].fillna(0)
```
#### 3.2 Drop Kolom
Drop feature 'City or Regency' dan 'Special Status'
```
df.drop(columns=cols_drop, inplace=True)
```
Mengecek perubahan yang dilakukan
```
df.isnull().sum()
```
![image](https://user-images.githubusercontent.com/74480780/126044064-6592ccea-edd7-453e-929c-b947f3fc7c3a.png)
![image](https://user-images.githubusercontent.com/74480780/126044075-8735d61e-e17d-422f-8b59-ccca07f335c8.png)

#### 3.3 Ubah Tipe Data Kolom Kategori Menjadi Numerik

Disini kita akan menggunakan teknik Label Encoding. Meskipun hampir semua kolom object dalam dataset tidak bersifat ordinal, namun jumlah kolom kategorik terlalu banyak sehingga akan terbentuk lebih banyak kolom saat kita melakukan Encoding dengan One-Hot Encoding yang tentu tidak efisien untuk dataset. Dengan pertimbangan tersebut, Label Encoding merupakan pilihan yang cocok.
```
# Membuat fungsi untuk mengubah kolom dengan metode Label Encoding
le = LabelEncoder()

def label_encoder(df):
  for col in df.columns:
    if df.dtypes[col] == 'object':
      le.fit(df[col].astype(str))
      df[col] = le.transform(df[col].astype(str))
  return df
```
```
# Terapkan fungsi pada dataset, lalu cek tipe data tiap kolom untuk melihat perubahannya
df = label_encoder(df)
df.info()
```
![image](https://user-images.githubusercontent.com/74480780/126044116-a3edabc7-f028-43ea-ade4-780c5270743d.png)
![image](https://user-images.githubusercontent.com/74480780/126044124-f57a6f9a-5cfc-4483-80f5-1b41efe38671.png)

### 4. Modeling
Membuat model machine learning yang akan memprediksi angka 'New Cases' dengan metode Decision Tree
#### 4.1 Splitting Data
Melakukan splitting data dengan fungsi `train_test_split()` dari library Scikit-learn
```
X = df.drop(columns=['New Cases'])
y = df['New Cases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
```
```
print(f"X train: {X_train.shape}")
print(f"y train: {y_train.shape}")
print(f"X test: {X_test.shape}")
print(f"y test: {y_test.shape}")
```
![image](https://user-images.githubusercontent.com/74480780/126044197-db25ef1c-f879-4551-82e7-9e586e393654.png)

#### 4.2 Training Model
```
model_dt = DecisionTreeClassifier().fit(X_train, y_train)
```
```
# cek akurasi
score_model_dt= model_dt.score(X_test, y_test)
print(f"Akurasi Decision Tree Model:\t{round(score_model_dt * 100, 3)}%")
```
![image](https://user-images.githubusercontent.com/74480780/126044233-dbd017fe-8e9e-45ea-9e0c-bba9a598796c.png)

### 5 Model Evaluation
#### 5.1 Improve Model with Feature Scaling
##### 5.1.1 Melakukan Feature scaling / Normalisasi Data dengan Fungsi `MinMaxScaler()` dari Library Skicit-learn
```
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
```
# Ubah data yang telah dinormalisasi ke dalam bentuk Dataframe
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
```
![image](https://user-images.githubusercontent.com/74480780/126044266-76f220b5-33b9-4565-9e25-5d1fb0ebc724.png)
##### 5.1.2 Melakukan Pemodelan Kembali dengan Data yang Sudah di Normalisasi
```
model_dt = DecisionTreeClassifier().fit(X_train_scaled, y_train)
```
```
score_scaled_model_dt = model_dt.score(X_test_scaled, y_test)
print(f"Akurasi Decision Tree Model yang Sudah di Normalisasi:\t{round(score_scaled_model_dt * 100, 3)}%")
```
![image](https://user-images.githubusercontent.com/74480780/126044395-4dd032a9-581f-46cb-8eed-5415ae4cf107.png)
Akurasi model dengan data yang telah dinormalisasi meningkat meskipun hanya berselisih kurang lebih satu angka dengan akurasi model sebelumnya.

### 6. Predict Test Data
#### 6.1 Melakukan Prediksi Dengan Data Test
```
Y_predict = model_dt.predict(X_test_scaled)
```
```
Y_predict
```
![image](https://user-images.githubusercontent.com/74480780/126044492-12e9de2d-7194-4095-b300-b5b1e40738f6.png)

#### 6.2 Mengubah Data Hasil Prediksi ke dalam Bentuk Dataframe
```
new_cases_prediction = pd.DataFrame({'New Cases': Y_predict})
```
```
new_cases_prediction.head(10)
```
![image](https://user-images.githubusercontent.com/74480780/126044517-c0601052-1244-468e-9ab8-186a33c57d62.png)

#### 6.3 Eksport DataFrame ke dalam bentuk file csv
```
filename = 'new_cases_prediction.csv'

new_cases_prediction.to_csv(filename, index=False)
print(f"File '{filename}' has been exported")
```
![image](https://user-images.githubusercontent.com/74480780/126044540-74bb1505-adff-413b-b903-b5f3ae5fd7bf.png)

---
## Project 2: Data Analysis [COVID-19 Indonesia Dataset (Analysis)]
Overview:
1. Membuat analisis data dengan python berdasarkan dataset yang berisi data COVID-19 di Indonesia
2. Dataset berasal dari [kaggle.com](https://www.kaggle.com/) dengan nama COVID-19 Indonesia Dataset yang disusun oleh Hendratno yang mengambil data dari beberapa sumber yaitu [situs resmi pemerintah SATGAS COVID-19](https://covid19.go.id/), [Badan Pusat Statistik](https://www.bps.go.id/), dan [Hub InaCOVID-19](https://bnpb-inacovid19.hub.arcgis.com/)
3. Dataset disusun berdasarkan time series atau sususan waktu, tingkat nasional, dan tingkat provinsi, juga beserta data demografi dari lokasi/daerah tersebut
4. Dataset memiliki 37 kolom
   - **'Date'** (Tanggal dilaporkan)
   - **'Location ISO Code'** (Kode lokasi berdasarkan standar ISO)
   - **'Location'** (Nama lokasi)
   - **'New Cases'** (Kasus positif harian)
   - **'New Deaths'** (Kasus kematian harian)
   - **'New Daily Recovered'** (Kasus kesembuhan harian)
   - **'New Active Cases'** (Kasus aktif harian)
   - **'Total Cases'** (Jumlah akumulatif kasus positif sampai waktu terkait)
   - **'Total Deaths'** (Jumlah akumulatif kasus kematian sampai waktu terkait)
   - **'Total Recovered'** (Jumlah akumulatif kasus kesembuhan sampai waktu terkait)
   - **'Total Active Cases'** (Jumlah akumulatif kasus aktif sampai waktu terkait)
   - **'Location Level'** (Tingkat lokasi regional atau nasional)
   - **'City or Regency'** (Nama kota atau wilayah)
   - **'Province'** (Nama provinsi lokasi)
   - **'Country'** (Nama negara lokasi)
   - **'Island'** (Nama pulau utama lokasi)
   - **'Time Zone'** (Zona waktu lokasi)
   - **'Special Status'** (Status istimewa lokasi)
   - **'Total Regencies'** (Jumlah kabupaten dalam lokasi terkait)
   - **'Total Cities'** (Jumlah kota dalam lokasi terkait)
   - **'Total Districts'** (Jumlah kecamatan dalam lokasi terkait)
   - **'Total Urban Village'** (Jumlah pedesaan dalam lokasi terkait)
   - **'Total Rural Village'** (Jumlah perkampungan dalam lokasi terkait)
   - **'Area (km2)'** (Area lokasi dalam kilometer persegi)
   - **'Population'** (Jumlah populasi dalam lokasi terkait)
   - **'Population Density'** (Kepadatan penduduk dalam lokasi terkait, rumus = Population / Area)
   - **'Longitude'** (Garis bujur lokasi)
   - **'Latitude'** (Garis lintang lokasi)
   - **'New Cases per Million'** (Rumus = (New Cases / Population) x 1.000.000)
   - **'Total Cases per Million'** (Rumus = (Total Cases / Population) x 1.000.000)
   - **'Total Deaths per Million'** (Rumus = (Total Deaths / Population) x 1.000.000)
   - **'Case Fatality Rate'** (Rumus = (Total Deaths / Total Cases) x 100)
   - **'Case Recovered Rate'** (Rumus = (Total Recovered / Total Cases) x 100)
   - **'Growth Factor of New Cases'** (Kurang dari 1 artinya menurun, 1 artinya tidak ada perubahan, lebih dari 1 artinya meningkat, rumus = Today New Cases / Yesterday New Cases)
   - **'Growth Factor of New Deaths'** (Kurang dari 1 artinya menurun, 1 artinya tidak ada perubahan, lebih dari 1 artinya meningkat, rumus = Today New Deaths / Yesterday New Deaths)
5. Tahapan dalam menganalisa data yang akan dilakukan terhadap dataset terbagi menjadi tiga tahap
   - Data Preparation
   - Data Wrangling
   - Data Analysis
6. Project menggunakan dataset berasal [kaggle](https://www.kaggle.com/), yang disusun oleh Hendratno.
   - Repository project Github dapat diakses [disini](https://github.com/hibartaufik/Data-Analysis-Covid-in-Indonesia)
   - Dataset dapat diakses [disini](https://www.kaggle.com/hendratno/covid19-indonesia)

### 1. Data Preparation
#### 1.1 Import Libraries
Import beberapa library yang dibutuhkan
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```
#### 1.2 Import Dataset
Import dataset yang akan digunakan, lalu tampilkan dataset untuk mengecek apakah import berhasil atau tidak
```
df = pd.read_csv('./dataset/covid_19_indonesia_time_series_all.csv')
df.head()
```
![image](https://user-images.githubusercontent.com/74480780/124726805-4bee4080-df38-11eb-8a99-0632f0fdc24e.png)
### 2. Data Wrangling
#### 2.1 Memahami Isi Data
Sumber dataset berasal dari kaggle.com, disusun oleh Hendratno. Data disusun berdasarkan time series, baik di tingkat negara (Indonesia), maupun di tingkat provinsi. Data didapatkan dari berbagai sumber yaitu [situs resmi pemerintah SATGAS COVID-19](https://covid19.go.id/), [Badan Pusat Statistik](https://www.bps.go.id/), dan [Hub InaCOVID-19](https://bnpb-inacovid19.hub.arcgis.com/).

Memeriksa dan menemukan data yang bersifat redundant, data yang terduplikasi, dan data NULL yang sekiranya tidak dibutuhkan dalam analisa. Setelah data ditemukan, maka data akan di drop/dibuang agar dataset yang di-analisa lebih rapi dan menghasilkan insight-insight yang lebih akurat. Hal pertama yang dilakukan ialah memahami data dari tiap kolom dengan membaca deskripsi atau summary tiap kolom dari dataset. Untuk dataset yang sedang digunakan, kita dapat mengetahui deskripsi tiap kolom dengan mengakses sumber dataset di [kaggle.com](https://www.kaggle.com/).

#### 2.2 Menemukan dan Membuang Data yang Tidak Dibutuhkan
- Memeriksa data tiap kolom
```
df.info()
```
![image](https://user-images.githubusercontent.com/74480780/124795772-9c868d80-df7a-11eb-87cb-bd0a1a2dc444.png)
![image](https://user-images.githubusercontent.com/74480780/124795898-c3dd5a80-df7a-11eb-9218-21e87d89db55.png)
Dataset memiliki 37 kolom, terdiri dari 12 kolom dengan tipe data object/string, 12 kolom dengan tipe data integer, dan 13 kolom dengan tipe data float. Kita juga dapat memeriksa dimensi dataset untuk memastikan jumlah kolom dan baris.
```
df.shape
```
![image](https://user-images.githubusercontent.com/74480780/124796275-2df5ff80-df7b-11eb-9e15-839ef8ea4431.png)
Mengetahui karakteristik data dengan memeriksa jumlah data NULL dalam setiap kolom
```
df.isnull().sum()
```
![image](https://user-images.githubusercontent.com/74480780/124796852-d1471480-df7b-11eb-95ed-e592f8c7243a.png)
![image](https://user-images.githubusercontent.com/74480780/124796942-e91e9880-df7b-11eb-9fb9-80f977cb3ff3.png)
Terlihat dalam beberapa kolom dataset memiliki data NULL, bahkan untuk kolom **'City or Regency'** semua datanya bernilai NULL. Setelah kita memahami isi data dengan membaca deskripsi tiap kolom, kita dapat menetukan kolom mana saja yang tidak terlalu penting dan harus di drop.

- Drop kolom yang dirasa tidak dibutuhkan

Kolom yang akan di drop yaitu **'Location ISO Code'**, **'City or Regency'**, **'Country'**, **'Continent'**, **'Time Zone'**, **'Special Status'**, **'Total Cities'**, **'Total Districts'**, **'Total Regencies'**, **'Total Urban Villages'**, **'Total Rural Villages'**, **'Area (km2)'**, **'New Cases per Million'**, **'Total Cases per Million'**, **'New Deaths per Million'**, **'Total Deaths per Million'**, **'Growth Factor of New Cases'**, dan **'Growth Factor of New Deaths'**.
```
df.drop(['Location ISO Code', 'City or Regency', 'Country', 'Continent', 'Time Zone', 'Special Status', 'Total Cities', 'Total Districts', 'Total Regencies', 'Total Urban Villages', 'Total Rural Villages', 'Area (km2)', 'New Cases per Million', 'Total Cases per Million', 'New Deaths per Million', 'Total Deaths per Million', 'Growth Factor of New Cases', 'Growth Factor of New Deaths'], axis=1, inplace=True)
```

- Ubah tipe data Date menjadi tipe data datetime agar menghindari kesalahan dalam pengurutan time-series untuk visualisasi

```
df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y")
```

Cek untuk melihat perubahan yang sudah dilakukan
```
df.info()
```
![image](https://user-images.githubusercontent.com/74480780/127913832-164d33c1-339d-4aa5-8340-fe2aff324f99.png)

Kini dataset memiliki 19 kolom
### 3. Data Analysis
Melihat korelasi atau hubungan tiap kolom pada dataset
```
# melihat korelasi/hubungan tiap data antar feature
df.corr()
```
![image](https://user-images.githubusercontent.com/74480780/124797820-e96b6380-df7c-11eb-80c3-2912285909af.png)
Agar mudah dipahami, korelasi/hubungan tiap kolom dapat dilihat melalui visualisasi heatmap
```
#visualisasi heatmap hubungan antar feature
mask = np.triu(df.corr())
plt.style.use('seaborn')
fg, ax = plt.subplots(figsize=(18, 7), dpi=200)
sns.heatmap(df.corr(), mask=mask, cmap='Reds', annot=True, linewidths=1)
plt.title('KORELASI TIAP FEATURE', fontweight='bold', fontsize=20)
plt.xticks(rotation=30)
plt.show()
```
![image](https://user-images.githubusercontent.com/74480780/124798163-41a26580-df7d-11eb-809f-7592305e9daf.png)

#### 3.1 Q1: Per tanggal berapa kasus baru COVID-19 terbanyak ditemukan dalam satu hari?
A1.1 : Urutan 10 tanggal dengan kasus COVID-19 paling banyak

Filter data berdasarkan kolom **'New Cases'**, lalu tampilkan 10 data dengan jumlah angka kesembuhan terbanyak beserta tanggalnya(kolom **'Date'**). Lalu tampilkan juga kolom **'Location'** untuk menunjukkan bahwa angka **'New Cases'** di-akumulatif tidak dari satu lokasi saja (Akumulatif dari seluruh wilayah di Indonesia per tanggal terkait).
```
top_10_date_new_cases = df.sort_values(by='New Cases', ascending=False, ignore_index=True)[['Date', 'New Cases', 'Location']][:10]
top_10_date_new_cases
```
![image](https://user-images.githubusercontent.com/74480780/125093059-29a52000-e0fc-11eb-9640-14f00cb48efa.png)
Berdasarkan urutan data di atas, kasus COVID-19 paling banyak ditemukan dalam satu hari yaitu pada tanggal 30 Januari 2021 sebanyak 14.518 kasus baru. Hal menariknya yaitu 9 dari 10 tanggal teratas dengan kasus COVID-19 per hari terbanyak berada pada bulan Januari 2021, artinya kasus cenderung naik pada bulan Januari 2021 dan mencapai puncaknya pada akhir bulan.

A1.2 : Visualisasi kenaikan **'New Cases'** berdasarkan urutan 10 tanggal dengan kasus COVID-19 paling banyak per hari
```
# mengakses data dan memasukannya ke dalam list agar dapat di-visualisasikan
x1_2 = list(top_10_date_new_cases.sort_values(by='Date').to_dict()['Date'].values())
y1_2 = list(top_10_date_new_cases.sort_values(by='Date').to_dict()['New Cases'].values())
```
```
plt.style.use('seaborn')
fg, ax = plt.subplots(figsize=(16,6), dpi=200)
sns.lineplot(x=x1_2, y=y1_2, color='blue', marker='o', linewidth=1)
plt.ylim(ymin=12000, ymax=15000)
plt.text(0.64, 0.81, 'Puncak kasus COVID-19 per hari sebanyak 14.518', fontsize=12,transform=fg.transFigure, color='red')
plt.title('Laju Pertumbuhan Kasus COVID-19 Per Tanggal 15 Januari - 6 Februari 2021', fontsize=18, fontweight='bold', pad=40)
plt.xlabel('Tanggal', fontsize=12, fontweight='bold')
plt.ylabel('Jumlah Kasus', fontsize=12, fontweight='bold')
plt.show()
```
![image](https://user-images.githubusercontent.com/74480780/124799193-7a8f0a00-df7e-11eb-9768-98738b5add1d.png)
Berdasarkan visualisasi di atas, pertumbuhan kasus dimulai semenjak pertengahan bulan. Lalu sempat naik turun hingga mencapai kasus minimal pada 23 Januari, namun justru setelah itu peningkatan kasus COVID-19 per hari semakin melonjak hingga mencapai puncaknya pada 30 Januari sebanyak 14.518 kasus per hari.
#### 3.2 Q2: Provinsi mana saja yang memiliki kasus baru terbanyak per hari?
A2.1: Urutan 5 provinsi dengan penemuan kasus COVID-19 per hari terbanyak

Filter data dari kolom **'New Cases'**, lalu tampilkan beserta provinsinya(kolom **'Province'**).
```
df_not_ll = df.loc[df['Location Level'] == 'Province'].sort_values(by='New Cases', ascending=False)[['Province', 'New Cases']]
df_not_ll.groupby('Province')[['New Cases']].sum().sort_values(by='New Cases', ascending=False)[:5]
```
![image](https://user-images.githubusercontent.com/74480780/124799935-47994600-df7f-11eb-8c0a-799802d428ec.png)
Provinsi DKI Jakarta memiliki total kasus per hari tertinggi sebanyak 379.204 kasus, Diikuti oleh Jawa Barat, Jawa Tengah, Jawa Timur, dan Kalimantan Timur. Angka ini didapat dengan menjumlahkan kasus per hari dari tiap provinsi, lalu mengurutkan 5 provinsi dengan jumlah kasus per hari tertinggi.

A2.2: Visualisasi perbandingan antar provinsi dengan total penemuan kasus COVID-19 per hari terbanyak
```
a22 = df_not_ll.groupby('Province')[['New Cases']].sum().sort_values(by='New Cases', ascending=False)[:5]
x2_2 = list(a22.to_dict()['New Cases'].keys())
y2_2 = list(a22.to_dict()['New Cases'].values())
```
```
plt.style.use('seaborn')
fg, ax = plt.subplots(figsize=(16,6), dpi=200)
sns.barplot(x=x2_2, y=y2_2, color='blue')
plt.ylim(ymax=450000)
plt.title('Perbandingan Tiap Provinsi dengan Total Penemuan Kasus Baru COVID-19 per Hari', fontsize=18, fontweight='bold', pad=40)
plt.xlabel('Provinsi', fontsize=12, fontweight='bold')
plt.ylabel('Jumlah Kasus', fontsize=12, fontweight='bold')
plt.show()
```
![image](https://user-images.githubusercontent.com/74480780/124800182-8cbd7800-df7f-11eb-8358-2e8f8266dec6.png)

#### 3.3 Q3: Pulau mana saja yang memiliki kasus per hari terbanyak? 
A3.1: Pulau dengan jumlah kasus per hari terbanyak

Filter data dari kolom **'New Cases'**, lalu tampilkan beserta pulaunya(kolom **'Island'**).
```
top_islands_new_cases = df.loc[df.Island != None].groupby(['Island'])[['New Cases']].sum().sort_values(by='New Cases', ascending=False)
top_islands_new_cases
```
![image](https://user-images.githubusercontent.com/74480780/124800563-fb9ad100-df7f-11eb-85b9-299c50e22e30.png)

A3.2: Visualisasi perbandingan tiap pulau dengan jumlah kasus per hari terbanyak
```
x3_2 = list(top_islands_new_cases.to_dict()['New Cases'].keys())
y3_2 = list(top_islands_new_cases.to_dict()['New Cases'].values())
```
```
plt.style.use('seaborn')
fg, ax = plt.subplots(figsize=(16,4), dpi=200)
sns.barplot(x=x3_2, y=y3_2, color='blue')
plt.title('Perbandingan Tiap Pulau dengan Total Penemuan Kasus Baru COVID-19 per Hari', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Pulau', fontsize=12, fontweight='bold')
plt.ylabel('Jumlah Kasus (Juta)', fontsize=12, fontweight='bold')
labels, location = plt.yticks()
plt.yticks(labels, (labels/1000000).astype(float))
plt.show()
```
![image](https://user-images.githubusercontent.com/74480780/124803211-0440d680-df83-11eb-8a36-b292d7a35390.png)
Berdasarkan visualisasi di atas total penemuan kasus COVID-19 per hari berpusat di pulau Jawa menembus hingga 1 juta kasus, perbandingan yang terlihat sangat mencolok jika dibandingkan dengan total penemuan kasus COVID-19 per hari di pulau lain. Hal ini cocok dengan analisa sebelumnya dimana 4 dari 5 provinsi dengan total penemuan kasus COVID-19 per hari terbanyak berada di pulau Jawa.

#### 3.4 Q4: Provinsi manakah dengan total rata-rata kesembuhan paling tinggi per hari?
A4.1: Urutan provinsi dengan kesembuhan paling tinggi per hari

Filter data dari kolom **'New Recovered'**, lalu tampilkan beserta provinsinya(kolom **'Island'**).
```
top_province_new_rcvry = df.loc[df['Location Level'] == 'Province'].groupby(['Province'])[['New Recovered']].sum().sort_values(by='New Recovered', ascending=False)
top_province_new_rcvry[:10]
```
![image](https://user-images.githubusercontent.com/74480780/124800952-74019200-df80-11eb-97f0-0619499991e4.png)

A4.2: Visualisasi perbandingan provinsi dengan kesembuhan tertinggi per hari
```
x4_2 = list(top_province_new_rcvry[:10].to_dict()['New Recovered'].keys())
y4_2 = list(top_province_new_rcvry[:10].to_dict()['New Recovered'].values())
```
```
plt.style.use('seaborn')
fg, ax = plt.subplots(figsize=(16,6), dpi=200)
sns.barplot(x=x4_2, y=y4_2, color='blue')
plt.title('Perbandingan Provinsi dengan Total Rata-Rata Kesembuhan Tertinggi per hari', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Provinsi', fontsize=12, fontweight='bold')
plt.ylabel('Jumlah Kesembuhan', fontsize=12, fontweight='bold')
plt.ylim(ymax=400000)
plt.show()
```
![image](https://user-images.githubusercontent.com/74480780/124801311-e70b0880-df80-11eb-95ea-cce6ec413e6c.png)
Visualisasi di atas menunjukkan bahwa DKI Jakarta unggul dalam angka kesembuhan per hari sebanyak 365.561, hampir dua kali lipat dari provinsi Jawa Barat yang menempati posisi kedua yaitu sebanyak 218.851.

#### 3.5 Q5: Time Series Based Visualization in Big Picture
Melihat peningkatan angka kasus baru, kematian, kesembuhan, dan kasus aktif dalam rentan waktu Maret 2020 - Maret 2021
```
data_plot = df.groupby('Date').agg({'New Cases':'sum', 'New Deaths':'sum', 'New Recovered':'sum', 'New Active Cases':'sum'})
```
```
fg, ax = plt.subplots(figsize=(12, 6), dpi=800)
plt.plot(data_plot)
plt.title('Time Series Based Visualization (Maret 2020 - Maret 2021)', fontsize=18, fontweight='bold', pad=20)
plt.show()
```
![q5picture](https://user-images.githubusercontent.com/74480780/127877423-adb6a0d6-2c17-4539-8829-bb6359b0ac06.png)


---
## Project 1: Stroke Prediction [Waroenk Skill Bootcamp Competition on Kaggle]
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

### 1. Data Preparation
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

### 2. Exploratory Data Analysis (EDA)
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
   
10. Melihat jumlah data pada data yang akan diprediksi (target/label)
    ```
    train['stroke'].value_counts()
    ```
   ![image](https://user-images.githubusercontent.com/74480780/110506191-74546280-8131-11eb-8c61-cd2828095246.png)
   
### 3. Data Preprocessing
Hal-hal yang ditemukan pada tahap exploratory data analysis yang perlu pengolahan data agar mendapatkan data yang ideal untuk membuat model machine learning
- Seimbangkan jumlah data target yang mengidap stroke (1) dan yang tidak (0)
- Ubah kolom dengan data yang bertipe object menjadi numerik

1. Seimbangkan jumlah data target yang mengidap stroke (1) dan yang tidak (0)
   ```
   train['stroke'].value_counts()
   ```
   ![image](https://user-images.githubusercontent.com/74480780/110507336-7ec32c00-8132-11eb-95d8-5e542190ab56.png)

   ```
   #pisahkan data target yang mengidap stroke dengan yang tidak ke dalam variabel yang berbeda
   negatif = train.loc[train['stroke'] == 0]
   positif = train.loc[train['stroke'] == 1]

   print(f"Jumlah Data Negatif:\t{len(negatif)}")
   print(f"Jumlah Data Positif:\t{len(positif)}")
   ```

   Menyeimbangkan jumlah data dengan menyamakan data negatif dengan data positif karena perbandingan data yang jauh akan lebih baik dilakukan dengan metode Undersampling. Lakukan undersampling dengan menyamakan jumlah data negatif yang jauh lebih banyak dengan jumlah data positif.

   ```
   negatif = negatif[:len(positif)]

   #cek kembali jumlah data target
   print(f"Jumlah Data Negatif:\t{len(negatif)}")
   print(f"Jumlah Data Positif:\t{len(positif)}")

   #gabungkan data negatif dengan positif
   new_data = pd.concat([negatif, positif], ignore_index=True)
   ```

2. Ubah kolom dengan data yang bertipe object/string menjadi tipe data numerik
   
   Terdapat dua metode untuk mengubah data yang bertipe object/string menjadi tipe data numerik, yaitu Label Encoding dan One Hot Encoding. Label Encoding dilakukan pada data yang memiliki tingkatan atau peringkat, sedangkan One Hot Encoding dilakukan pada data yang tidak memiliki tingkatan apapun. 

   Berdasarkan karakteristik data, metode yang akan digunakan ialah One Hot Encoding karena data yang diubah tipenya tidak memiliki tingkatan atau peringkat. Lakukan metode One Hot Encoding menggunakan fungsi get_dummies pada library pandas.
   ```
   #lakukan One Hot Encoding pada data yang sudah diseimbangkan
   new_data = pd.get_dummies(new_data, drop_first=True)
   #lakukan One Hot Encoding pada data yang tidak diseimbangkan
   train = pd.get_dummies(train, drop_first=True)
   #lakukan One Hot Encoding pada data test juga
   test = pd.get_dummies(test, drop_first=True)
   ```
   Saat data dicek kembali, terlihat data yang asalnya bertipe object/string sudah berubah menjadi data yang bertipe numerik
   ![image](https://user-images.githubusercontent.com/74480780/110509242-7bc93b00-8134-11eb-8fbe-30061c30bb67.png)

### 4. Create Machine Learning Model
Setelah data diolah dan dirasa telah ideal, maka selanjutnya ialah membuat model machine learning dari dataset tersebut. Berdasarkan studi kasus dan karakteristik data target, metode yang akan digunakan adalah klasifikasi dengan Decision Tree. Mengapa klasifikasi? karena tujuan dibuatnya model machine learning ini adalah untuk memprediksi pasien yang positif (1) mengidap stroke dan yang tidak mengidap (0) stroke, artinya model bertujuan untuk mengelompokkan (klasifikasi) pasien ke dalam dua buah golongan, yaitu yang mengidap stroke dan yang tidak mengidap stroke. Dengan begitu, model akan dibuat dengan DecisionTreeClassifier() pada library sklearn.tree.

Terdapat dua buah model machine learning yang akan dibuat. Model pertama adalah model dengan data yang sudah diseimbangkan jumlah datanya, sedangkan model kedua ialah model dengan data yang tidak diseimbangkan. Pada tahap Model Evaluation, kedua model ini akan dibandingkan bagaimana peforma nilai akurasinya untuk memprediksi data target dengan berbagai metode evaluasi.

1. Pisahkan data terlebih dahulu menjadi data feature dan target
   ```
   #lakukan pada data yang diseimbangkan (new_data)
   X = new_data.drop(['id_pasien', 'stroke'], axis=1)
   y = new_data['stroke']

   #lakukan pada data yang tidak diseimbangkan (train)
   X_pure = train.drop(['id_pasien', 'stroke'], axis=1)
   y_pure = train['stroke']
   ```
   
2. Pisahkan data untuk 70% melatih data dan 30% untuk testing
   ```
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
   X2_train, X2_test, y2_train, y2_test = train_test_split(X_pure, y_pure, test_size=0.3, stratify=y_pure)
   ```
3. Buat model machine learning dengan Decision Tree
   ```
   #buat dan pasangkan model dengan data train yang telah diseimbangkan
   model_dt = DecisionTreeClassifier().fit(X_train, y_train)
   #buat dan pasangkan model dengan data train yang tidak diseimbangkan
   model_dt_pure = DecisionTreeClassifier().fit(X2_train, y2_train)
   ```
4. Lakukan pengecekan akurasi dengan fungsi score() 
   ```
   #mengecek akurasi model machine learning yang telah dibuat dengan data test
   
   #model dengan data yang diseimbangkan
   score_1 = model_dt.score(X_test, y_test)
   #model dengan data yang tidak diseimbangkan
   score_2 = model_dt_pure.score(X_test, y_test)
   
   print(f"Akurasi Model 1: {round(score_1 * 100, 2)}%")
   print(f"Akurasi Model 2: {round(score_2 * 100, 2)}%")
   ```
   ![image](https://user-images.githubusercontent.com/74480780/110517891-04000e00-813e-11eb-8942-58390aed82d9.png)
   
### 5. Model Evaluation
Selain pengecekan akurasi dengan fungsi score(), dilakukan juga pengecekan dengan menggunakan metric lain dengan fungsi classification_report() pada library sklearn.metrics
1. Pengecekan akurasi dengan classification_report()
   ```
   #melakukan pengecekan peforma dengan classification_report()

   #pengecekan pada model dengan data yang diseimbangkan
   print("BALANCED DATA")
   print(classification_report(y_test, model_dt.predict(X_test)))
   #pengecekan pada model dengan data yang tidak diseimbangkan
   print("UNBALANCE DATA")
   print(classification_report(y2_test, model_dt_pure.predict(X2_test)))
   ```
   ![image](https://user-images.githubusercontent.com/74480780/110514819-44f62380-813a-11eb-944d-519ff85e4d71.png)
   Agar lebih jelas, dilakukan juga pengecekan dengan confusion matrix beserta visualisasi dengan heatmap
2. Pengecekan akurasi dengan confusion_matrix dan visualisasi dengan heatmap
   - Lakukan pada model machine learning dari data yang di seimbangkan
   ```
   plt.style.use('ggplot')
   fg, ax = plt.subplots(figsize=(12, 6))
   mx1 = confusion_matrix(y_test, model_dt.predict(X_test))
   sns.heatmap(mx1, cmap='Reds', annot=True, linewidths=2)
   plt.title("PENGECEKAN CONFUSION MATRIX", pad=20, fontsize=20, fontweight='bold')
   plt.show()

   #berdasarkan visualisasi sebelumnya dapat dilihat presentase-nya

   print("Model dengan data yang diseimbangkan")
   print(f"TRUE POSITIF: {round(mx1[0][0] / (mx1[0][0] + mx1[0][1]) * 100, 2)}%")
   print(f"TRUE NEGATIF: {round(mx1[1][1] / (mx1[1][1] + mx1[1][0]) * 100, 2)}%")
   ```
   ![image](https://user-images.githubusercontent.com/74480780/110515665-31978800-813b-11eb-8fcf-bc6aa142696f.png)

   Berdasarkan visualisasi di atas dapat dilihat bahwa model dengan data yang diseimbangkan memiliki presentase
   ![image](https://user-images.githubusercontent.com/74480780/110515748-4d9b2980-813b-11eb-95f2-f5343c8dd35e.png)
   
   - Lakukan pada model machine learning dari data yang tidak diseimbangkan
   ```
   plt.style.use('ggplot')
   fg, ax = plt.subplots(figsize=(12, 6))
   mx2 = confusion_matrix(y_test, model_dt_pure.predict(X_test))
   sns.heatmap(mx2, cmap='Reds', annot=True, linewidths=2)
   plt.title("PENGECEKAN CONFUSION MATRIX", pad=20, fontsize=20, fontweight='bold')
   plt.show()
   
   #berdasarkan visualisasi sebelumnya dapat dilihat presentase-nya

   print("Model dengan data yang tidak diseimbangkan")
   print(f"TRUE POSITIF: {round(mx2[0][0] / (mx2[0][0] + mx2[0][1]) * 100, 2)}%")
   print(f"TRUE NEGATIF: {round(mx2[1][1] / (mx2[1][1] + mx2[1][0]) * 100, 2)}%")
   ```
   ![image](https://user-images.githubusercontent.com/74480780/110516327-07929580-813c-11eb-8a73-5a2e73046e17.png)

   Berdasarkan visualisasi di atas dapat dilihat bahwa model dengan data yang tidak diseimbangkan memiliki presentase
   ![image](https://user-images.githubusercontent.com/74480780/110516417-2002b000-813c-11eb-974b-66a8f785f705.png)
   
   Kesimpulan yang dapat diambil berdasarkan pengecekan akurasi dengan confusion matrix di atas adalah kita dapat mengetahui perbandingan jumlah TRUE POSITIF, TRUE NEGATIF,      FALSE POSITIF, dan FALSE NEGATIF dari kedua buah model. Berdasarkan studi kasus kali ini, model yang memprediksi lebih banyak pasien yang stroke (TRUE POSITIF)
   lebih baik karena artinya model dapat memprediksi kecenderungan pasien yang memiliki peluang besar mengidap stroke walau sebenarnya dia didiagnosa belum/tidak mengidap stroke.
   
### 6. Predict Test Data
Sekarang, kedua model sudah layak untuk dapat melakukan prediksi yang akan menghasilkan kumpulan data berbentuk list. Karena prediksi ini akan dikumpulkan di kaggle.com, maka perlu dilakukan perubahan bentuk dimensi agar sesuai dengan format data yang diminta.
1. Sesuaikan bentuk data dengan drop 'id_pasien'
   ```
   #drop 'id_pasien' terlebih dahulu
   new_test = test.drop('id_pasien', axis=1)
   ```
2. Melakukan prediksi dan memasukkan data prediksi tersebut ke dalam variabel
   ```
   #melakukan prediksi pada model dengan data yang diseimbangkan, lalu masukan ke variabel baru
   predict = model_dt.predict(new_test)

   #melakukan prediksi pada model dengan data yang tidak diseimbangkan, lalu masukan ke variabel baru
   predict_pure = model_dt_pure.predict(new_test)
   ```
3. Membuat dataframe yang akan dikumpulkan di kaggle.com
   ```
   #lakukan pada prediksi dari model dengan data yang diseimbangkan
   collect_1 = pd.DataFrame()
   collect_1['id_pasien'] = test['id_pasien']
   collect_1['stroke'] = predict

   #lakukan pada prediksi dari model dengan data yang tidak diseimbangkan
   collect_2 = pd.DataFrame()
   collect_2['id_pasien'] = test['id_pasien']
   collect_2['stroke'] = predict_pure
   ```
4. Export Kedua dataframe tersebut ke dalam file yang berformat csv (.csv)
   ```
   #export dataframe yang berisi prediksi
   collect_1.to_csv('collect1.csv', index=False)
   collect_2.to_csv('collect2.csv', index=False)
   ```
### Kesimpulan
Lalu pertanyaannya, model mana yang lebih baik memprediksi orang yang mengidap stroke atau tidak? meskipun dalam beberapa pengecekan akurasi model dengan data yang tidak diseimbangkan memiliki angka yang lebih baik, namun hal tersebut bukan berarti model tersebut lebih baik. Wajar jika unbalance model memprediksi orang yang mengidap stroke lebih banyak karena model tersebut memang dibuat dan dipasangkan menggunakan data train yang memiliki data dengan label positif lebih banyak. Sedangkan untuk balanced model, skor akurasinya lebih kecil namun seimbang dalam distribusi jumlah labelnya. Dengan begitu, kedua model ini sama-sama dapat digunakan tergantung bagaimana kebutuhan dan situasinya.
