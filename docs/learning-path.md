# Jalur belajar

Proyek ini lebih mudah dipahami jika Anda berhenti setelah tiap tahap dan menebak bentuk tensor berikutnya sebelum menjalankan kode. Gunakan `python3 -m unittest discover -s tests -v` untuk memeriksa implementasi dasar.

## 1. Gambar dan patch

Buka `patchify()` dan `unpatchify()` di `src/vit_scratch/model.py`. Masukan model berukuran `[B, 1, 28, 28]`. Patch 4 × 4 menghasilkan kisi 7 × 7, sehingga keluarannya `[B, 49, 16]`.

**Latihan:** Bayangkan gambar 8 × 8 satu kanal dengan patch 4 × 4. Ada berapa patch dan berapa angka dalam tiap patch? Temukan patch kedua pada gambar itu.

**Periksa:** Ada empat patch, masing-masing berisi 16 angka. Patch kedua berada di kanan atas. Tes `test_patch_order_and_reconstruction` memeriksa urutan yang sama pada gambar 28 × 28.

## 2. Token dan attention

`PatchEmbedding` memproyeksikan 16 piksel menjadi 64 angka. `TinyViT` menambahkan satu token kelas dan positional embedding, sehingga bentuknya `[B, 50, 64]`. Dalam tiap head, lebar 64 dibagi empat menjadi 16. Matriks attention per head berukuran `[50, 50]`.

**Latihan:** Pada matriks itu, apa arti baris ke-0 dan kolom ke-0? Mengapa jumlah satu baris mendekati 1?

**Periksa:** Baris ke-0 berisi bobot dari query token kelas ke semua token. Kolom ke-0 adalah key token kelas. `softmax` diterapkan sepanjang kolom key, sehingga jumlah bobot dalam tiap baris adalah 1. Tes `test_logits_attention_and_gradients` memeriksanya.

## 3. Training

Jalankan satu langkah training pada tes lebih dulu. Perhatikan urutan `zero_grad`, forward, cross-entropy, `backward`, lalu `step` di `src/vit_scratch/train.py`. Setelah itu jalankan latihan penuh. Data resmi 60.000 gambar dibagi 54.000 train dan 6.000 validasi dengan seed 42. Data uji resmi 10.000 gambar dibaca untuk evaluasi setelah checkpoint terbaik dipilih.

**Latihan:** Mengapa checkpoint dipilih dari loss validasi, bukan akurasi uji?

**Periksa:** Data uji dipakai untuk mengukur hasil akhir. Memilih checkpoint darinya akan membuat keputusan training bergantung pada data yang seharusnya tidak terlihat oleh proses pemilihan model.

## 4. Baca hasil eksperimen

Bandingkan `artifacts/vit_metrics.json` dengan `artifacts/cnn_metrics.json`. Lihat akurasi uji, jumlah parameter, waktu latihan, kurva validasi, dan `confusion_matrix`. Matriks itu memakai baris label asli dan kolom prediksi.

**Latihan:** Cari pasangan kelas yang paling sering tertukar dalam masing-masing model. Apakah satu run cukup untuk menyimpulkan bahwa arsitektur tertentu selalu unggul?

**Periksa:** Ambil sel terbesar di luar diagonal, lalu sebutkan label baris dan kolomnya. Satu run hanya menggambarkan konfigurasi dan seed ini; variasi seed atau tuning dapat mengubah hasil.

## 5. Jelajahi web

Di web, pilih gambar, token yang bertanya, lapis, dan head. Amati bahwa peta berubah. Saat token kelas dipilih, angka di peta hanya mencakup 49 patch; bobot ke token kelas ditulis terpisah. Warna dinormalisasi untuk tiap pilihan, jadi warna dua peta yang berbeda tidak bisa dibandingkan sebagai angka absolut.

**Latihan:** Pilih satu gambar dan dua head pada lapis yang sama. Catat patch dengan bobot tertinggi di masing-masing head. Apakah ini membuktikan bahwa patch tersebut menyebabkan prediksi?

**Periksa:** Tidak. Bobot attention menunjukkan cara token menggabungkan informasi pada head itu. Pengaruh terhadap keluaran juga bergantung pada value, proyeksi, residual, MLP, dan lapis lain.
