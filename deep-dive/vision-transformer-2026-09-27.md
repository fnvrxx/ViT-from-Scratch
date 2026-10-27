# Dari gambar sampai halaman web

Panduan ini mengikuti data yang sama dari berkas Fashion-MNIST, melalui training, sampai menjadi tampilan interaktif. Baca bersama [jalur belajar](../docs/learning-path.md), lalu jalankan tes atau training setelah tiap bagian.

## 1. Masalah yang diselesaikan

Fashion-MNIST berisi gambar abu-abu 28 × 28 piksel dengan sepuluh label pakaian. Model menerima tensor `[B, 1, 28, 28]` dan mengeluarkan sepuluh logit per gambar. `B` adalah jumlah gambar dalam satu batch. Logit adalah skor sebelum softmax; `argmax` memilih kelas dengan skor tertinggi.

ViT dalam proyek ini berukuran kecil supaya setiap tahapnya dapat dibaca dan dilatih di CPU. Paper aslinya melaporkan kekuatan ViT setelah pretraining pada data besar. Eksperimen ini tidak mencoba menyamai ukuran atau hasil tersebut.

## 2. Aliran data, bentuk, dan alasan desain

| Tahap | Bentuk untuk satu gambar | Alasan |
| --- | --- | --- |
| Gambar | `[1, 28, 28]` | Satu kanal intensitas abu-abu. |
| Patch | `[49, 16]` | Gambar dibagi menjadi kisi 7 × 7, tiap bagian 4 × 4. |
| Patch embedding | `[49, 64]` | Proyeksi linear memberi tiap patch ruang representasi 64 dimensi. |
| Token dan posisi | `[50, 64]` | Token kelas ditambahkan, lalu posisi yang dipelajari menjaga identitas letak patch. |
| Attention tiap head | `[50, 50]` | Tiap query memberi bobot pada seluruh key, termasuk dirinya dan token kelas. |
| Keluaran encoder | `[50, 64]` | Residual menjaga jalur informasi melalui attention dan MLP. |
| Logit kelas | `[10]` | Vektor token kelas dibaca oleh classifier. |

### Patchify

Di [`model.py`](../src/vit_scratch/model.py), `unfold` membentuk potongan tanpa mengubah nilai piksel. `permute` menyusun patch dalam urutan baris, lalu `reshape` meratakan setiap patch. Fungsi `unpatchify` membalik urutan ini. Pasangan fungsi tersebut dan tes rekonstruksi menjaga satu kontrak penting: patch nomor 1 di web benar-benar patch kiri atas yang dibaca model.

Alternatifnya adalah `Conv2d` dengan kernel dan stride sebesar patch. Itu dapat menghitung proyeksi patch dengan efisien, tetapi operasi `unfold` dan `Linear` membuat hubungan antara 16 piksel dan satu token lebih mudah dilihat pada proyek belajar ini.

### Proyeksi, token kelas, dan posisi

`PatchEmbedding` memakai `nn.Linear(16, 64)`. Semua patch berbagi bobot proyeksi yang sama. `TinyViT` menambahkan parameter token kelas di awal urutan dan positional embedding berukuran `[1, 50, 64]`. Posisi perlu diberikan karena attention sendiri menerima urutan token dan tidak mengetahui koordinat dua dimensi patch.

Token kelas berperan sebagai tempat model mengumpulkan representasi untuk klasifikasi. Alternatif yang masuk akal adalah merata-ratakan seluruh token patch setelah encoder. Proyek ini memakai token kelas agar dekat dengan alur ViT yang dijelaskan pada paper asli dan agar query token kelas dapat diperiksa di web.

### Self-attention yang ditulis sendiri

`MultiHeadSelfAttention` memproyeksikan setiap token menjadi query `Q`, key `K`, dan value `V`. Dengan empat head, setiap head mendapat lebar 16. Skor satu head dihitung sebagai `QKᵀ / √16`, lalu `softmax` sepanjang dimensi key menghasilkan bobot. Bobot tersebut menggabungkan `V`, empat hasil head disatukan, kemudian diproyeksikan kembali ke lebar 64.

Untuk batch, bentuk bobotnya `[B, 4, 50, 50]`. Baris adalah token yang bertanya, kolom adalah token yang dibaca. Karena `softmax` ada pada sumbu kolom, setiap baris berjumlah 1. Kode memakai perkalian matriks PyTorch dan autograd, tetapi tidak memakai `nn.MultiheadAttention` atau encoder siap pakai. Ini menjaga perhitungan yang ingin dipelajari tetap terlihat.

`EncoderBlock` memakai LayerNorm sebelum attention dan sebelum MLP. Setiap subbagian memiliki residual: `tokens + update`. MLP menggunakan dua linear layer dengan GELU di tengah. Dua block menjaga waktu CPU terjangkau. Keluaran token kelas dari block terakhir masuk ke LayerNorm dan classifier.

## 3. Dari dataset ke checkpoint

[`data.py`](../src/vit_scratch/data.py) mengambil train dan test resmi dari TorchVision. `random_split` dengan seed 42 membagi 60.000 gambar train menjadi 54.000 train dan 6.000 validasi. `ToTensor` mengubah piksel menjadi rentang 0 sampai 1; tidak ada augmentasi atau normalisasi tambahan.

[`train.py`](../src/vit_scratch/train.py) mengacak batch train, menghitung cross-entropy, melakukan backpropagation, lalu memperbarui bobot dengan AdamW. Setelan awalnya batch 128, learning rate 0,0003, paling banyak 10 epoch, dan penghentian jika loss validasi tidak membaik tiga epoch berturut-turut. Checkpoint terbaik dipilih berdasarkan loss validasi. Setelah training selesai, kode memuat checkpoint itu dan mengevaluasinya sekali pada test resmi.

Hasil menyimpan riwayat train dan validasi, akurasi uji, confusion matrix, jumlah parameter, serta durasi training. Baris confusion matrix adalah label asli, kolomnya prediksi. Durasi bergantung pada mesin. Seed membantu reproduksi, tetapi beberapa detail komputasi dan versi library dapat membuat hasil di mesin lain sedikit berbeda.

[`cnn.py`](../src/vit_scratch/cnn.py) menjadi pembanding kecil dengan dua convolution dan pooling. Pembagian data, optimizer, learning rate, batch, dan batas epoch sama. Arsitektur berbeda, sehingga perbandingan ini adalah titik awal yang jujur, bukan eksperimen pencarian hyperparameter menyeluruh. Satu run per model tidak cukup untuk menyimpulkan pemenang umum.

## 4. Dari checkpoint ke web

[`export_demo.py`](../src/vit_scratch/export_demo.py) memuat checkpoint ViT terbaik dan mengambil contoh uji pertama dari setiap kelas. Indeks dipilih berdasarkan urutan data, bukan menurut apakah model benar. Pada `eval()` dan `inference_mode()`, skrip menyimpan piksel, label asli, probabilitas, serta bobot attention lengkap untuk setiap lapis dan head. Data dan metrik kedua model ditulis ke `web/public/data/demo.json`.

[`main.js`](../web/src/main.js) memuat JSON itu di browser. Pilihan gambar mengganti semua panel. Pilihan query, lapis, dan head mengambil satu baris dari `attention[layer][head][query][key]`. Query 0 adalah token kelas; query 1 sampai 49 adalah patch. Kolom 0 juga token kelas. Peta hanya menggambar kolom 1 sampai 49 sebagai kisi 7 × 7 dan menulis bobot kolom 0 secara terpisah.

Warna peta dibagi oleh bobot patch terbesar pada baris yang sedang ditampilkan. Ini membuat pola mudah dibaca, tetapi berarti merah pekat pada dua pilihan berbeda belum tentu memiliki bobot numerik yang sama. Kode menulis bobot terbesar sebagai angka agar pembaca tidak bergantung pada warna saja. Web tidak menjalankan model lagi, sehingga kontrol tidak mengubah prediksi. Ia memilih bagian dari rekaman inferensi yang sama.

Attention menunjukkan distribusi bobot pada satu head, bukan pengukuran sebab-akibat pada prediksi. Value, proyeksi, residual, MLP, dan lapis berikutnya turut memengaruhi hasil. Karena itu web memakai istilah “bobot attention”, tidak “alasan model memilih kelas”.

## 5. Cara memeriksa pemahaman

1. Ubah sebuah piksel pada tensor sebelum `patchify`. Periksa bahwa hanya satu patch hasil mentah yang berubah.
2. Cetak bentuk `Q`, `K`, `V`, skor, dan bobot untuk batch dua gambar. Cocokkan dengan tabel di atas.
3. Pilih dua head pada gambar yang sama di web. Bandingkan nilai numeriknya, lalu jelaskan mengapa warna saja tidak cukup.
4. Buka kedua berkas metrik dan temukan kesalahan terbesar di luar diagonal confusion matrix. Tulis pasangan label yang tertukar tanpa menyebutnya sebagai kesimpulan umum.

## Bacaan lanjutan

- [Paper Vision Transformer](https://arxiv.org/abs/2010.11929) untuk arsitektur dan konteks pretraining.
- [PyTorch: tensor dan autograd](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) untuk operasi dan gradien yang dipakai di sini.
- [PyTorch: dataset dan DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) untuk cara batch dibentuk.
- [Repositori Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) untuk sumber data dan label.
- [Attention is not Explanation](https://arxiv.org/abs/1902.10186) untuk batas penggunaan bobot attention sebagai penjelasan.
- [Panduan situs statis Vite](https://vite.dev/guide/static-deploy) untuk langkah publikasi ketika situs siap dibagikan.
