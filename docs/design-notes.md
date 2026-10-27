# Arah visual web

Pilihan pengguna: editorial dengan rasa jurnal sains. Pembacanya adalah orang yang baru mempelajari ViT dan peninjau portfolio yang ingin memeriksa apakah visual cocok dengan komputasi model.

Desain dibaca sebagai **ENERGY 2 / RHYTHM 2 / MOTION 1**. Judul serif besar memberi karakter artikel; teks antarmuka sans-serif menjaga label kontrol tetap ringkas. Latar seperti kertas dan tinta gelap memberi ruang bagi gambar kecil, peta attention, dan angka eksperimen.

Merah bata dipakai untuk query atau pilihan aktif, bobot attention, dan seri ViT pada grafik. Hubungan warna itu membantu pembaca mengikuti bagian model yang sedang diperiksa. Warna gelap membedakan seri CNN. Kisi yang berulang berasal dari pembagian gambar menjadi 7 × 7 patch; ia juga menjadi batas patch yang dapat diklik, bukan pola latar dekoratif.

Label kecil bernomor menandai bab dan dataset tanpa mengambil perhatian dari judul. Gradasi pada legenda heatmap menunjukkan rentang bobot, sedangkan bayangan hanya muncul pada bingkai gambar utama supaya ia terasa seperti sampel yang diletakkan di atas meja. Panah pada tautan utama menandai perpindahan ke meja eksperimen.

Susunan halaman mengikuti tugas pembaca: pahami perubahan bentuk tensor, jelajahi keluaran model, lalu baca hasil dan batasnya. Gerak hanya muncul saat pilihan berubah atau saat halaman menggulir ke bagian yang dituju. Semua angka metrik dibaca dari hasil training, sedangkan gambar demo berasal dari data uji Fashion-MNIST.

Warna teks utama terhadap latar kertas memiliki rasio 13,16:1. Teks aksen terhadap kertas 5,84:1 dan terhadap latar lab 5,10:1. Tepi kontrol `#74817b` terhadap panel `#fffdf7` memiliki rasio 3,99:1. Rasio tersebut dihitung dengan pemeriksa kontras pada skill `antislop-human`.
