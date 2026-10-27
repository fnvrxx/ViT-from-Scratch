# Audit antislop, 27 September 2026

Status berikut diperiksa pada web hasil build dengan data model yang diekspor. Tes otomatis: empat tes Python lulus, `npm run build` lulus, dan `tests/browser_smoke.py` lulus di Chromium headless. Screenshot desktop dan ponsel diperiksa secara visual.

## Hard Gate

- R-02 PASS: pencarian pada teks web tidak menemukan em dash.
- R-03 PASS: lebar 320, 375, dan 400 piksel tidak menimbulkan overflow; screenshot ponsel menunjukkan konten tetap terbaca.
- R-17 PASS: angka model berasal dari berkas metrik training; ukuran gambar dan kelas berasal dari konfigurasi serta dataset.
- R-18 PASS: halaman tidak memuat testimonial atau orang rekaan.
- R-23 PASS: gambar berasal dari Fashion-MNIST; visual lain adalah diagram data dan kisi patch yang diminta untuk menjelaskan ViT, tanpa logo atau avatar rekaan.
- R-24 PASS: tes browser memeriksa target seluruh tautan internal.
- R-25 PASS: pasangan teks yang dipakai lulus 4,5:1; aksen pada latar lab 5,10:1 dan tepi kontrol pada panel 3,99:1.
- R-26 PASS: tes browser memakai semua tahap, sepuluh contoh, query, layer, head, klik patch, dan tombol retry.
- R-27 PASS: teks loading, keadaan kosong, dan kesalahan memiliki sebab serta langkah berikutnya; tes browser mencoba kedua tombol retry.
- R-28 PASS: halaman tidak memakai FAQ pengisi.
- R-32 PASS: Tab pertama mencapai skip link dengan outline; tombol tahap bekerja dengan Enter, dan pilihan model memakai kontrol native.
- R-33 PASS: perilaku dan gaya ditulis langsung di `web/src`, tanpa skrip yang menambal sumber dari luar.
- R-34 PASS: hanya satu tema jurnal sains yang dikirim; tidak ada toggle tema yang setengah jadi.
- R-35 PASS: build produksi lulus; tes browser mengklik kontrol dan memeriksa desktop, ponsel, serta kesalahan JavaScript.
- R-36 PASS: halaman tidak memakai klaim keamanan, pelanggan, atau kinerja yang dibuat-buat.
- R-37 PASS: arah “jurnal sains” dipilih pengguna sebelum implementasi dan dicatat bersama dials di `docs/design-notes.md`.
- R-38 PASS: gambar, label, prediksi, dan metrik berasal dari dataset serta run yang nyata.

## Purpose-Gate

- R-01 PASS: gradasi hanya menandai rentang bobot pada legenda heatmap; alasannya tertulis di catatan desain.
- R-04 PASS: tidak ada paket ikon generik; wordmark memakai teks dan simbol fungsi hanya menandai navigasi atau rumus.
- R-06 PASS: serif besar memberi rasa artikel ilmiah; label kecil bernomor mengurutkan bab, sesuai catatan desain.
- R-07 PASS: kisi adalah pembagian 7 × 7 patch dari model dan menjadi batas pilihan pada gambar.
- R-08 PASS: satu panah pada tautan utama menandai perpindahan ke eksperimen; panah lain menyatakan aliran tensor.
- R-09 PASS: tidak ada badge atau pill dekoratif.
- R-10 PASS: tidak ada glassmorphism.
- R-12 PASS: bayangan hanya pada sampel gambar utama untuk kesan lembar gambar di atas meja.
- R-13 PASS: tidak ada glow pada kontrol, kartu, atau latar.
- R-14 PASS: panel gambar, attention, dan prediksi mempunyai isi serta peran berbeda, bukan kartu fitur seragam.
- R-19 PASS: gerak mengikuti pilihan pembaca; scroll halus dimatikan saat `prefers-reduced-motion` aktif.
- R-22 PASS: tidak ada ilustrasi generik; gambar berasal dari sampel data uji.

## Liveliness

- Dials PASS: ENERGY 2, RHYTHM 2, MOTION 1 tercatat di catatan desain.
- Konsistensi PASS: tata letak artikel bervariasi antara pembuka, alur, meja eksperimen, hasil, dan catatan; gerak tetap rendah.
- Fokus PASS: judul menjadi fokus pembuka, gambar dan peta menjadi fokus eksperimen, angka uji menjadi fokus evaluasi.
- Ruang PASS: jarak memisahkan bab dan mengelompokkan kontrol dengan visual yang dipengaruhinya.
- Aksen PASS: merah bata menghubungkan pilihan aktif, bobot attention, dan seri ViT; warna lain memberi dasar dan pembanding CNN.
- Motif PASS: kisi patch berulang pada gambar, sampel, dan visualisasi attention.
- Design Read PASS: arah editorial jurnal sains berasal dari pilihan pengguna dan dijabarkan sebelum perancangan visual final.

## Craftsmanship dan Quality Locks

- C-1 PASS: keputusan visual utama memiliki alasan singkat di catatan desain.
- C-2 PASS: tes browser menjalankan kontrol dan memeriksa responsnya.
- C-3 PASS: setiap bagian menjawab tugas pembaca: alur tensor, keluaran model, hasil, atau batas interpretasi.
- C-4 PASS: ukuran ponsel, navigasi keyboard, data kosong, data gagal, dan retry diuji.
- C-5 PASS: metrik dan sampel berasal dari training serta test resmi; tidak ada bukti sosial rekaan.
- R-05 PASS: susunan bab mengikuti materi ViT, tanpa template fitur, testimoni, atau pricing.
- R-11 PASS: kontrol bersudut tegas sesuai arah jurnal, bukan kumpulan pill seragam.
- R-15 PASS: tautan utama menyebut tindakan “Buka meja eksperimen”.
- R-16 PASS: teks web tidak memakai jargon pemasaran AI.
- R-20 PASS: gambar Fashion-MNIST dan kisi patch memberi identitas khusus pada proyek ini.
- R-21 PASS: tema terang dipilih untuk arah kertas jurnal; tidak ada dark mode yang dipaksakan.
- R-29 PASS: kertas, tinta, dan satu aksen merah bata membentuk palet utama; hijau gelap membedakan seri CNN.
- R-30 PASS: komposisi artikel dan meja eksperimen berasal dari isi, bukan salinan tata letak produk lain.
- R-31 PASS: alasan warna, tipografi, kisi, bayangan, gradasi, dan gerak tercatat di catatan desain.

## Copy dan komentar kode

- `$antislop-copywriting` PASS: teks web memakai angka bersumber, istilah konkret, CTA spesifik, dan tanpa em dash atau klaim pelanggan.
- `$antislop-code` PASS: kode tidak berisi komentar dekoratif, narasi tiap baris, atau TODO samar; docstring hanya menjelaskan kontrak yang perlu diketahui.
- `$antivibe` PASS: `deep-dive/vision-transformer-2026-09-27.md` mengikuti aliran data, alasan desain, alternatif, latihan, dan bacaan primer dari awal sampai web.
