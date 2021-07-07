# Hibar_Portfolio
Data science portfolio

## Project 2: Data Analysis [COVID-19 Indonesia Dataset]
Overview:
1. Membuat analisis data dengan python berdasarkan dataset yang berisi data COVID-19 di Indonesia
2. Dataset berasal dari [kaggle.com](https://www.kaggle.com/) dengan nama COVID-19 Indonesia Dataset yang disusun oleh Hendratno yang mengambil data dari beberapa sumber yaitu [situs resmi pemerintah SATGAS COVID-19](https://covid19.go.id/), [Badan Pusat Statistik](https://www.bps.go.id/), dan [Hub InaCOVID-19](https://bnpb-inacovid19.hub.arcgis.com/)
3. Dataset disusun berdasarkan time series atau sususan waktu, tingkat nasional, dan tingkat provinsi, juga beserta data demografi dari lokasi/daerah tersebut
4. Dataset memiliki 37 feature/kolom
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
5. Tahapan dalam menganalisa data terbagi menjadi dua tahap
   - Data Preparation
   - Exploratory Data Analysis
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
   
### 2. Exploratory Data Analysis (EDA)
#### 2.1 Memahami Isi Data
- Melihat 5 data teratas
  ```
  #highlight 5 data teratas
  df.head()
  ```
  ![image](https://user-images.githubusercontent.com/74480780/124726805-4bee4080-df38-11eb-8a99-0632f0fdc24e.png)
- Melihat tipe data tiap kolom/feature
  ```
  df.info()
  ```
  ![image](https://user-images.githubusercontent.com/74480780/124729949-15fe8b80-df3b-11eb-8d6e-5ec7ded8a21c.png)
  ![image](https://user-images.githubusercontent.com/74480780/124730067-34fd1d80-df3b-11eb-80fe-1dbf24a61be2.png)
  Dataset memiliki 37 feature/kolom, terdiri dari 12 feature dengan tipe data numerikal dan 25 feature dengan tipe data numerikal(12 kolom dengan tipe data integer dan 13 kolom dengan tipe data float)

- Melihat dimensi data dari dataset
  ```
  df.shape
  ```
  ![image](https://user-images.githubusercontent.com/74480780/124730542-a76dfd80-df3b-11eb-9483-cefc2eaddb11.png)

- Melihat summary statistics tiap kolom dari dataset
  ```
  df.describe().T
  ```
  ![image](https://user-images.githubusercontent.com/74480780/124730967-02075980-df3c-11eb-9f7b-261d924b143d.png)
  ![image](https://user-images.githubusercontent.com/74480780/124731085-1ba8a100-df3c-11eb-8dec-6e7507344003.png)

- Melihat data NULL dalam dataset
  ```
  df.isnull().sum()
  ```
  ![image](https://user-images.githubusercontent.com/74480780/124731338-56aad480-df3c-11eb-8d2a-46bb3a20bf22.png)
  ![image](https://user-images.githubusercontent.com/74480780/124731437-6e825880-df3c-11eb-98a0-0e0dbc8a10bd.png)
      
- Melihat korelasi/hubungan antar kolom/feature
  ```
  df.corr()
  ```
  ![image](https://user-images.githubusercontent.com/74480780/124731731-b30df400-df3c-11eb-98b6-231469bed61b.png)
  Agar lebih mudah melihat korelasi tiap kolom/feature, maka solusinya dengan melakukan visualisasi dengan heatmap
  ```
  #visualisasi heatmap hubungan antar feature
  mask = np.triu(df.corr())

  plt.style.use('seaborn')
  fg, ax = plt.subplots(figsize=(22, 12), dpi=200)
  sns.heatmap(df.corr(), mask=mask, cmap='Reds', annot=True, linewidths=1)
  plt.title('KORELASI TIAP FEATURE', fontweight='bold', fontsize=20)
  plt.xticks(rotation=30)
  plt.show()
  ```
  ![image](https://user-images.githubusercontent.com/74480780/124737278-fe76d100-df41-11eb-8356-2834bcecadb6.png)

- Menambahkan kolom tahun pada dataset
  ```
  # menambahkan kolom tahun
  df['Year'] = df['Date'].str[-4:]
  df['Year'] = df['Year'].astype('int32')
  ```
#### 2.2 Analisa Data
   
- Q1: Per tanggal berapa kasus baru COVID-19 terbanyak ditemukan dalam satu hari?

  A1.1 : Urutan 10 tanggal dengan kasus COVID-19 paling banyak
  ```
  top_10_date_new_cases = df.sort_values(by='New Cases', ascending=False, ignore_index=True)[['Date', 'New Cases']][:10]
  top_10_date_new_cases
  ```
  ![image](https://user-images.githubusercontent.com/74480780/124733189-282df900-df3e-11eb-9c80-04a444a11bf6.png)
  Berdasarkan urutan data di atas, kasus COVID-19 paling banyak ditemukan dalam satu hari yaitu pada tanggal 30 Januari 2021 sebanyak 14.518 kasus baru. Hal menariknya yaitu 9 dari 10 tanggal teratas dengan kasus COVID-19 per hari terbanyak berada pada bulan Januari 2021, artinya kasus cenderung naik pada bulan Januari 2021 dan mencapai puncaknya pada akhir bulan.
   
  A1.2 : Visualisasi kenaikan 'New Cases' berdasarkan urutan 10 tanggal dengan kasus COVID-19 paling banyak per hari
  ```
  # memasukkan dataset urutan tanggal dengan kasus tertinggi ke dalam variabel baru
  top_10_date_new_cases = df.sort_values(by='New Cases', ascending=False, ignore_index=True)[['Date', 'New Cases']][:10]
  # mengakses data dan memasukannya ke dalam list agar dapat di-visualisasikan
  x = list(top_10_date_new_cases.sort_values(by='Date').to_dict()['Date'].values())
  y = list(top_10_date_new_cases.sort_values(by='Date').to_dict()['New Cases'].values())
  ```
  ```
  # visualisasi dengan lineplot
  plt.style.use('seaborn')
  fg, ax = plt.subplots(figsize=(16,6), dpi=200)
  sns.lineplot(x=x, y=y, color='blue', marker='o', linewidth=1)
  plt.ylim(ymin=12000)
  labels, locations = plt.yticks()
  plt.yticks(labels, (labels/1000).astype(float))
  plt.text(0.64, 0.81, 'Puncak kasus COVID-19 per hari sebanyak 14.518', fontsize=12,transform=fg.transFigure, color='red')
  plt.title('Laju Pertumbuhan Kasus COVID-19 Per Tanggal 15 Januari - 6 Februari 2021', fontsize=18, fontweight='bold', pad=40)
  plt.xlabel('Tanggal', fontsize=12, fontweight='bold')
  plt.ylabel('Jumlah Kasus (Ribu)', fontsize=12, fontweight='bold')
  plt.show()
  ```
  ![image](https://user-images.githubusercontent.com/74480780/124734772-a6d76600-df3f-11eb-811c-0d88cc6d8ae9.png)
  Berdasarkan visualisasi di atas, pertumbuhan kasus dimulai semenjak pertengahan bulan. Lalu sempat naik turun hingga mencapai kasus minimal pada 23 Januari, namun justru setelah itu peningkatan kasus COVID-19 per hari semakin melonjak hingga mencapai puncaknya pada 30 Januari sebanyak 14.518 kasus per hari.
   
- Q2: Provinsi mana saja yang memiliki kasus baru terbanyak per hari?

  A2.1: Urutan 5 provinsi dengan penemuan kasus COVID-19 per hari terbanyak
  ```
  df_not_ll = df.loc[df['Location Level'] == 'Province'].sort_values(by='New Cases', ascending=False)[['Province', 'New Cases']]
  df_not_ll.groupby('Province')[['New Cases']].sum().sort_values(by='New Cases', ascending=False)[:5]
  ```
  ![image](https://user-images.githubusercontent.com/74480780/124735487-485eb780-df40-11eb-82a9-b46fdf20c02e.png)
  Provinsi DKI Jakarta memiliki total kasus per hari tertinggi sebanyak 379.204 kasus, Diikuti oleh Jawa Barat, Jawa Tengah, Jawa Timur, dan Kalimantan Timur. Angka ini didapat dengan menjumlahkan kasus per hari dari tiap provinsi, lalu mengurutkan 5 provinsi dengan jumlah kasus per hari tertinggi.
   
  A2.2: Visualisasi perbandingan antar provinsi dengan total penemuan kasus COVID-19 per hari terbanyak
  ```
  a22 = df_not_ll.groupby('Province')[['New Cases']].sum().sort_values(by='New Cases', ascending=False)[:5]
  x = list(a22.to_dict()['New Cases'].keys())
  y = list(a22.to_dict()['New Cases'].values())
  ```
  ```
  plt.style.use('seaborn')
  fg, ax = plt.subplots(figsize=(16,6), dpi=200)
  sns.barplot(x=x, y=y, color='blue')
  plt.ylim(ymax=450000)
  plt.title('Perbandingan Tiap Provinsi dengan Total Penemuan Kasus Baru COVID-19 per Hari', fontsize=18, fontweight='bold', pad=40)
  plt.xlabel('Provinsi', fontsize=12, fontweight='bold')
  plt.ylabel('Jumlah Kasus', fontsize=12, fontweight='bold')
  plt.show()
  ```
  ![image](https://user-images.githubusercontent.com/74480780/124735951-bb682e00-df40-11eb-8fc1-c587109750c1.png)
   
- Q3: Pulau mana saja yang memiliki kasus per hari terbanyak?

  A3.1: Pulau dengan jumlah kasus per hari terbanyak
  ```
  top_islands_new_cases = df.loc[df.Island != None].groupby(['Island'])[['New Cases']].sum().sort_values(by='New Cases', ascending=False)
  top_islands_new_cases
  ```
  ![image](https://user-images.githubusercontent.com/74480780/124736137-ec486300-df40-11eb-967b-3d6964a4fbda.png)
   
  A3.2: Visualisasi perbandingan tiap pulau dengan jumlah kasus per hari terbanyak
  ```
  x = list(top_islands_new_cases.to_dict()['New Cases'].keys())
  y = list(top_islands_new_cases.to_dict()['New Cases'].values())
  ```
  ```
  plt.style.use('seaborn')
  fg, ax = plt.subplots(figsize=(16,4), dpi=200)
  sns.barplot(x=x, y=y, color='blue')
  plt.title('Perbandingan Tiap Pulau dengan Total Penemuan Kasus Baru COVID-19 per Hari', fontsize=18, fontweight='bold', pad=20)
  plt.xlabel('Pulau', fontsize=12, fontweight='bold')
  plt.ylabel('Jumlah Kasus', fontsize=12, fontweight='bold')
  labels, location = plt.yticks()
  plt.yticks(labels, (labels/1000000).astype(float))
  plt.show()
  ```
  ![image](https://user-images.githubusercontent.com/74480780/124736461-3893a300-df41-11eb-8661-ed151f0d3639.png)
  Berdasarkan visualisasi di atas total penemuan kasus COVID-19 per hari berpusat di pulau Jawa menembus hingga 1 juta kasus, perbandingan yang terlihat sangat mencolok jika dibandingkan dengan total penemuan kasus COVID-19 per hari di pulau lain. Hal ini cocok dengan analisa sebelumnya dimana 4 dari 5 provinsi dengan total penemuan kasus COVID-19 per hari terbanyak berada di pulau Jawa.

- Q4: Provinsi manakah dengan total rata-rata kesembuhan paling tinggi per hari?

  A4.1: Urutan provinsi dengan total rata-rata kesembuhan paling tinggi per hari
  ```
  top_province_new_rcvry = df.loc[df['Location Level'] == 'Province'].groupby(['Province'])[['New Recovered']].sum().sort_values(by='New Recovered', ascending=False)
  top_province_new_rcvry[:10]
  ```
  ![image](https://user-images.githubusercontent.com/74480780/124736617-64af2400-df41-11eb-9de4-236a1c560c66.png)
   
  A4.2: Visualisasi perbandingan provinsi dengan total rata-rata kesembuhan tertinggi per hari
  ```
  x = list(top_province_new_rcvry[:10].to_dict()['New Recovered'].keys())
  y = list(top_province_new_rcvry[:10].to_dict()['New Recovered'].values())
  ```
  ```
  plt.style.use('seaborn')
  fg, ax = plt.subplots(figsize=(16,6), dpi=200)
  sns.barplot(x=x, y=y, color='blue')
  plt.title('Perbandingan Provinsi dengan Total Rata-Rata Kesembuhan Tertinggi per hari', fontsize=18, fontweight='bold', pad=20)
  plt.xlabel('Provinsi', fontsize=12, fontweight='bold')
  plt.ylabel('Jumlah Kesembuhan', fontsize=12, fontweight='bold')
  plt.ylim(ymax=400000)
  plt.show()
  ```
  ![image](https://user-images.githubusercontent.com/74480780/124736783-890b0080-df41-11eb-8352-61db10138b77.png)
   
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
   
10. Melihat jumlah data pada data feature yang akan diprediksi (target/label)
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
