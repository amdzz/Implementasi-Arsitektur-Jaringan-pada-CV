# Implementasi-Arsitektur-Jaringan-pada-CV

Proyek ini menunjukkan bagaimana penerapan model menggunakan salah satu penerapan arsitektur jaringan dalam visi komputer yakni custom CNN digunakan untuk klasifikasi gambar dengan dataset X-ray. Dataset yang digunakan bersifat publik dan diperoleh dari [kaggle](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data).

Untuk memverifikasi bahwa proses augmentasi dan pelabelan berjalan dengan benar, dilakukan visualisasi terhadap 9 sampel gambar hasil dari ImageDataGenerator. Gambar-gambar ini dipilih secara acak dari dataset pelatihan, kemudian ditampilkan dalam bentuk grid 3x3. Setiap gambar diberi label prediksi yang dihasilkan oleh model klasifikasi, yakni "Normal" untuk gambar tanpa patah tulang dan "Patah" untuk gambar yang menunjukkan indikasi patah tulang.

![image](https://github.com/user-attachments/assets/fc4fa3dd-e5bb-4e88-bd3d-59573e78ba22)

Gambar di atas memperlihatkan bahwa teknik augmentasi seperti rotasi, zoom, flipping, dan shifting telah berhasil diterapkan, sehingga menciptakan variasi data yang lebih luas tanpa mengubah makna dari label yang dimiliki. Augmentasi ini penting dilakukan agar model dapat belajar dari data yang beragam dan lebih tahan terhadap perbedaan posisi atau sudut pandang gambar saat digunakan di dunia nyata. Selain itu, visualisasi ini juga berguna untuk memastikan bahwa augmentasi tidak menyebabkan distorsi berlebihan yang bisa merusak struktur penting pada citra X-ray. Dengan demikian, proses augmentasi ini diharapkan dapat meningkatkan performa generalisasi model saat proses pelatihan.

## 1. Implementasi Arsitektur CNN

![image](https://github.com/user-attachments/assets/afe4e424-1d12-43e8-9456-61b8c8db0816)
(Source: https://www.upgrad.com/blog/basic-cnn-architecture/)

Model yang digunakan dalam implementasi pertama dibangun menggunakan arsitektur Convolutional Neural Network (CNN). Model diawali dengan layer konvolusi pertama yang memiliki 32 filter berukuran 3x3 dengan padding 'same', diikuti oleh aktivasi ReLU dan normalisasi batch untuk menstabilkan distribusi aktivasi selama pelatihan. Selanjutnya, dilakukan downsampling melalui MaxPooling 2x2 untuk mengurangi dimensi spasial gambar. Layer konvolusi kedua dan ketiga masing-masing memiliki 64 dan 128 filter, mengikuti pola yang sama: konvolusi → batch normalization → max pooling. Ketiga blok konvolusi ini berfungsi untuk mengekstraksi fitur spasial dan tekstur penting dari citra input secara bertahap.

Setelah fitur diekstraksi, layer Flatten digunakan untuk mengubah hasil konvolusi menjadi vektor satu dimensi, yang kemudian diproses oleh dua layer dense (fully connected) masing-masing dengan 256 dan 128 unit. Dropout diterapkan pada kedua layer ini untuk mengurangi risiko overfitting, dengan dropout rate masing-masing 0.5 dan 0.3. Terakhir, output layer terdiri dari satu neuron dengan aktivasi sigmoid, karena kasus ini merupakan klasifikasi biner antara citra bertulang Normal dan Patah. Model dikompilasi menggunakan optimizer Adam dengan learning rate yang relatif kecil (0.00009), menggunakan fungsi loss binary_crossentropy dan metrik akurasi. Pemilihan hyperparameter ini dilakukan dengan pertimbangan agar model dapat belajar secara stabil dan optimal terhadap dataset medis yang umumnya memiliki ukuran terbatas namun sangat penting akurasinya.

### Hasil

![image](https://github.com/user-attachments/assets/861d5463-ad97-4d7b-9b2c-eae3b854becb)

Pada grafik Training vs Validation Accuracy, terlihat bahwa akurasi model pada data pelatihan meningkat secara konsisten dari epoch ke-0 hingga sekitar epoch ke-8. Akurasi validasi juga menunjukkan tren serupa, dengan performa yang cukup stabil dan mendekati akurasi pelatihan, terutama setelah epoch ke-2. Sementara itu, grafik Training vs Validation Loss juga memperkuat kesimpulan tersebut. Loss pada data pelatihan dan validasi menurun tajam selama beberapa epoch pertama dan mulai mendatar di sekitar epoch ke-4, menandakan bahwa model dengan cepat belajar mengenali pola penting pada data. Secara keseluruhan, grafik ini menunjukkan bahwa arsitektur CNN yang digunakan berhasil mencapai performa yang baik.

![image](https://github.com/user-attachments/assets/88d8094b-5b23-4648-9c14-0442c357b83a)

Secara keseluruhan, model mencapai akurasi sebesar 90% pada data uji yang terdiri dari total 506 sampel. Nilai macro average dan weighted average untuk precision, recall, dan f1-score semuanya berada di angka 0.90–0.91, mengindikasikan bahwa model memiliki performa yang konsisten dan seimbang di kedua kelas, tanpa bias berlebihan terhadap satu kelas tertentu.

### Inferensi

![image](https://github.com/user-attachments/assets/1b97b5bb-01ad-471d-8215-0a7d91ea5714)

## 2. Implementasi Arsitektur VGG16

![image](https://github.com/user-attachments/assets/3189133c-3bbb-4e6f-895f-ca25b27d65d3)
(Source: https://www.interviewbit.com/blog/cnn-architecture/)

Pada tahap ini, arsitektur VGG16 digunakan sebagai dasar model klasifikasi citra, dengan pendekatan transfer learning. Dalam implementasi ini, bagian fully connected layers (classifier) dari VGG16 dihapus (include_top=False) agar hanya bagian feature extractor yang digunakan. Ukuran input disesuaikan menjadi (224, 224, 3) agar sesuai dengan standar input VGG16 dan dataset yang digunakan. Selanjutnya, lapisan output dari VGG16 di-flatten untuk diubah menjadi vektor satu dimensi, yang kemudian diteruskan ke dua lapisan Dense berturut-turut dengan masing-masing 512 unit dan fungsi aktivasi ReLU. Di antara kedua lapisan dense tersebut disisipkan Dropout sebesar 0.5 untuk mencegah overfitting. Akhirnya, ditambahkan lapisan output berupa neuron tunggal dengan fungsi aktivasi sigmoid, karena tugas klasifikasi ini bersifat biner (patah dan tidak patah).

Model kemudian di-compile menggunakan Adam optimizer dengan learning rate yang kecil (0.00009) agar pelatihan berlangsung stabil. Loss function yang digunakan adalah binary crossentropy, yang sesuai untuk klasifikasi dua kelas. Model ini juga mengunci semua bobot dari VGG16 bawaan (base_model.trainable = False) agar hanya classifier baru yang dilatih, sehingga menghemat waktu komputasi dan memanfaatkan fitur-fitur visual yang sudah dipelajari oleh VGG16 sebelumnya.

### Hasil

![image](https://github.com/user-attachments/assets/db238e52-f137-4907-b7e3-0bcbc52c6464)

Pada grafik akurasi (kiri), terlihat bahwa akurasi pelatihan meningkat tajam hingga mendekati 100%, sementara akurasi validasi juga mengalami peningkatan hingga mencapai sekitar 96%. Pola ini menunjukkan bahwa model mampu belajar dengan baik dari data pelatihan dan juga mampu melakukan generalisasi yang cukup baik terhadap data validasi. Sedangkan pada grafik loss (kanan), nilai loss pada data pelatihan menurun secara konsisten, menunjukkan bahwa model terus membaik dalam meminimalkan kesalahan prediksi. Secara keseluruhan, grafik ini menunjukkan bahwa penggunaan VGG16 dengan fine-tuned classifier menghasilkan model yang sangat baik, dengan akurasi tinggi dan loss yang rendah, serta stabilitas yang baik antara data pelatihan dan validasi. 

![image](https://github.com/user-attachments/assets/5c3cfb26-dbe0-4283-b369-d9116f4b9b26)

Secara keseluruhan, model mencetak akurasi total sebesar 0.99 atau 99%, yang berarti hanya sedikit kesalahan klasifikasi dari total 506 data uji. Nilai rata-rata (macro dan weighted average) untuk precision, recall, dan f1-score semuanya berada pada angka 0.99, mencerminkan konsistensi performa model di seluruh kelas. Dengan performa setinggi ini, dapat disimpulkan bahwa model VGG16 yang digunakan telah melakukan generalisasi dengan sangat baik dan layak digunakan untuk tugas klasifikasi citra X-ray patah tulang.

### Inferensi

![image](https://github.com/user-attachments/assets/de8961ec-1670-4f9c-8d15-3952561acb66)

## 3. Implementasi Arsitektur ResNet50

![image](https://github.com/user-attachments/assets/4b64f8a6-c6b0-4dbc-b34c-e198a55ba73c)
(Source: https://www.interviewbit.com/blog/cnn-architecture)

Arsitektur ResNet50 yang diimplementasikan menggunakan pendekatan transfer learning. Pada proses ini, bagian atas dari arsitektur ResNet50, yaitu lapisan fully connected (classifier), dihilangkan dengan mengatur parameter include_top=False. Model kemudian menerima input gambar berukuran 224x224 piksel dengan tiga kanal warna (RGB). Seluruh bobot dari ResNet50 dibekukan (trainable=False) agar tidak diperbarui selama pelatihan. Selanjutnya, di atas output dari ResNet50 ditambahkan beberapa lapisan custom classifier untuk menyesuaikan model dengan tugas klasifikasi dua kelas (binary classification). Output dari backbone terlebih dahulu diratakan menggunakan lapisan Flatten, kemudian dilanjutkan dengan dua buah lapisan dense berukuran 512 unit yang masing-masing diikuti oleh lapisan Dropout sebesar 0.5. Akhirnya, lapisan output menggunakan satu neuron dengan fungsi aktivasi sigmoid untuk menghasilkan probabilitas dari dua kelas. 

Model ini dikompilasi menggunakan optimizer Adam dengan learning rate yang kecil (0.00009), untuk memastikan proses pelatihan berlangsung stabil, dengan fungsi loss binary_crossentropy yang sesuai untuk klasifikasi biner. Secara keseluruhan, arsitektur ini menggabungkan kekuatan representasi fitur dari ResNet50 dengan fleksibilitas lapisan dense untuk menyesuaikan model dengan dataset X-ray yang digunakan.


### Hasil

![image](https://github.com/user-attachments/assets/0817c777-15fc-48bf-ab99-39e720602302)

Pada grafik akurasi, terlihat bahwa akurasi pelatihan mengalami peningkatan yang cukup stabil dari sekitar 0.53 hingga mencapai sekitar 0.70 pada akhir epoch. Sementara itu, akurasi validasi terlihat fluktuatif, namun secara umum tetap berada pada kisaran yang lebih tinggi dari akurasi pelatihan, mencapai puncaknya di atas 0.75. Pada grafik loss, baik loss pelatihan maupun loss validasi menunjukkan penurunan yang konsisten seiring bertambahnya epoch. Ini menunjukkan bahwa model belajar dengan baik dan tidak mengalami masalah overfitting yang signifikan, karena loss pada data validasi juga ikut menurun. Secara keseluruhan, grafik ini menunjukkan bahwa model mengalami proses pelatihan yang cukup stabil dengan peningkatan performa yang baik pada data pelatihan dan validasi.

![image](https://github.com/user-attachments/assets/2019d8ef-3c1a-4cb5-996a-b14ba86f7b9e)

Secara keseluruhan, akurasi model mencapai 0.71 atau 71%, artinya dari total 506 sampel, 71% diklasifikasikan dengan benar. macro average untuk precision, recall, dan f1-score adalah masing-masing 0.72, 0.71, dan 0.71. Rata-rata ini memberi bobot yang sama untuk setiap kelas tanpa memperhatikan ukuran kelas. Sementara itu, weighted average juga menghasilkan nilai yang sama, mencerminkan bahwa distribusi jumlah sampel antar kelas relatif seimbang. Dengan demikian, model menunjukkan performa yang cukup baik secara keseluruhan, meskipun masih terdapat ruang perbaikan khususnya dalam meningkatkan recall untuk kelas fractured, yang sangat penting dalam konteks deteksi kondisi medis seperti patah tulang.

### Inferensi

![image](https://github.com/user-attachments/assets/373c899c-48fa-4ce9-bc4f-67a8112fdfdb)

## 4. Implementasi Arsitektur YOLOv11

![image](https://github.com/user-attachments/assets/a1d8c14e-f12d-4a79-8c5b-0a01daedcca5)
(source: https://www.researchgate.net/figure/Shows-the-architecture-of-YoloV11_fig2_389021414)

Pada bagian ini dilakukan implementasi arsitektur YOLOv11, yaitu versi terbaru dari keluarga algoritma YOLO (You Only Look Once) yang dikenal sebagai metode deteksi objek real-time yang sangat cepat dan efisien. YOLOv11 merupakan pengembangan dari versi-versi sebelumnya, dengan berbagai peningkatan pada efisiensi inferensi, akurasi deteksi, dan kemampuan generalisasi terhadap berbagai skenario visual. Arsitektur ini memanfaatkan pendekatan end-to-end convolutional neural network yang langsung memprediksi bounding box dan kelas objek dalam satu kali proses, sehingga sangat cocok digunakan untuk aplikasi real-time seperti sistem diagnosis berbasis citra medis.

Dalam implementasi ini, model diinisialisasi dengan memuat bobot dari pre-trained model yolo11m.pt, yang merupakan varian medium dari YOLOv11 dengan keseimbangan antara ukuran dan performa. Model kemudian dilatih menggunakan dataset yang telah dikonfigurasi melalui file data.yaml, yang mendeskripsikan jalur data dan label kelas. Parameter pelatihan disesuaikan untuk keperluan eksperimen, di antaranya jumlah epoch sebanyak 10, ukuran gambar 640 piksel, dan batch size 16. Proses pelatihan dilakukan menggunakan GPU agar lebih efisien dan cepat, dengan mekanisme early stopping menggunakan parameter patience=5, yaitu menghentikan pelatihan lebih awal jika tidak terjadi peningkatan selama 5 epoch berturut-turut. Hasil pelatihan disimpan pada folder proyek runs/train dengan nama eksperimen exp.

### Hasil

![image](https://github.com/user-attachments/assets/45009411-627a-4af3-93cd-1ec6d001d1ff)

Grafik di atas merupakan hasil visualisasi metrik dan loss dari proses pelatihan model YOLOv11 selama 10 epoch. Terdapat 10 subplot yang menggambarkan performa model dari sisi loss dan metrik evaluasi, baik pada data train maupun val, serta performa deteksi berdasarkan precision, recall, dan mean Average Precision (mAP). Pada bagian atas grafik ditampilkan tiga jenis loss selama pelatihan: box_loss, cls_loss, dan dfl_loss. Ketiganya sempat mengalami peningkatan pada awal pelatihan, namun secara umum menunjukkan tren penurunan yang konsisten hingga akhir epoch ke-10. Ini menandakan bahwa model berhasil mempelajari fitur-fitur yang relevan dari data dan melakukan perbaikan secara bertahap. box_loss berkaitan dengan akurasi prediksi posisi bounding box, cls_loss mencerminkan kesalahan dalam klasifikasi objek, dan dfl_loss digunakan untuk prediksi distribusi posisi bounding box yang lebih akurat.

Grafik metrik precision(B) dan recall(B) menunjukkan peningkatan yang jelas, dengan precision mencapai nilai mendekati 0.9 dan recall sekitar 0.65 pada akhir pelatihan. Hal ini menandakan bahwa model mampu melakukan prediksi yang semakin tepat dan lengkap seiring bertambahnya epoch. Pada bagian bawah grafik, loss pada data validasi juga menunjukkan tren menurun yang signifikan. val/cls_loss mengalami penurunan paling tajam dari sekitar 17.5 ke hampir 1, menunjukkan perbaikan besar dalam klasifikasi objek di data validasi. Hal ini menjadi indikasi kuat bahwa model tidak hanya belajar dengan baik di data pelatihan, tetapi juga mampu melakukan generalisasi dengan baik ke data yang belum pernah dilihat.

Terakhir, dua grafik metrics/mAP50(B) dan metrics/mAP50-95(B) menunjukkan pertumbuhan signifikan. Nilai mAP50 meningkat hingga hampir 0.8, sementara mAP50-95 mencapai lebih dari 0.45. Ini menunjukkan bahwa model semakin baik dalam mendeteksi objek dengan akurasi tinggi dalam berbagai kondisi threshold Intersection over Union (IoU). Secara keseluruhan, grafik-grafik ini menunjukkan bahwa proses pelatihan model YOLOv11 berjalan dengan baik dan konsisten, dengan peningkatan performa pada semua aspek utama, baik pada data pelatihan maupun validasi. Tidak tampak adanya overfitting yang signifikan, dan model berhasil mencapai hasil deteksi yang cukup menjanjikan.

### Inferensi

![image](https://github.com/user-attachments/assets/c278530f-f54d-4ec6-9791-62bbd8cd4644)

Untuk hasil Inferensi pada video, dapat dilihat pada video berikut:

https://github.com/user-attachments/assets/cb073f4e-7900-4601-802e-bf1e224b8590

Kelompok 5 :
- Zoni Aryantoni Albab (G1A022043)
- Yebi Depriansyah (G1A022063)
- Ahmad Zul Zhafran (G1A022088)
