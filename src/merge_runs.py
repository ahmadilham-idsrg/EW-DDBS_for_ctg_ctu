"""
merge_runs.py
=============
Gabungkan beberapa direktori hasil (mis. seed 42 yang sudah jalan + seed 7 & 2024
yang baru) menjadi satu direktori siap dianalisis.

Aman dipakai karena berkas proba/ dan guide/ dinamai seed{S}_fold{F}.npz,
sehingga tidak pernah bertabrakan antar-seed.

Catatan penting untuk Google Colab
----------------------------------
/content DIHAPUS setiap kali runtime di-reset atau terputus. Jadi direktori
hasil dari sesi sebelumnya (mis. results_uci) TIDAK akan ada lagi di sesi baru,
meskipun Anda sudah mengunduh zip-nya ke komputer.

Skrip ini menanganinya dengan dua cara:
  1. Bila `results_uci/` hilang tetapi `results_uci.zip` ada di direktori kerja
     (atau di Google Drive yang ter-mount), arsip itu diekstrak otomatis.
  2. Bila benar-benar tidak ada, skrip berhenti SEBELUM membuat direktori
     keluaran, dan mencetak daftar apa saja yang sebenarnya tersedia.

Pemakaian
---------
    python merge_runs.py --out results_uci_all results_uci results_uci_s2
    python analyze_results.py --dir results_uci_all --ref "EWDDBS (Entropy+Safe+Tomek)"
"""

import argparse
import glob
import os
import shutil
import zipfile

import pandas as pd


#: tempat tambahan yang ikut dicari saat sebuah arsip .zip dilacak
ZIP_SEARCH_PATHS = [
    '.',
    '/content',
    '/content/drive/MyDrive',
    '/content/drive/MyDrive/ewddbs',
]


def _find_zip(name):
    """Cari <name>.zip di direktori kerja dan di Drive yang ter-mount."""
    for base in ZIP_SEARCH_PATHS:
        cand = os.path.join(base, f'{name}.zip')
        if os.path.exists(cand):
            return cand
    return None


def restore_if_needed(d):
    """Kembalikan direktori hasil dari arsip .zip bila direktorinya hilang.

    Ini kasus yang paling sering terjadi di Colab: runtime di-reset, /content
    kosong, tetapi pengguna masih punya results_uci.zip. Mengekstrak ulang jauh
    lebih baik daripada menjalankan ulang eksperimen 40 menit.
    """
    if os.path.isdir(d):
        return True

    z = _find_zip(os.path.basename(d.rstrip('/')))
    if z is None:
        return False

    print(f'  {d}/ tidak ada — memulihkan dari {z}')
    os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(d)

    # make_archive() mengarsipkan ISI folder, tetapi sebagian pengguna membuat
    # zip yang membungkus foldernya sekali lagi. Tangani kedua bentuk itu.
    if not os.path.exists(os.path.join(d, 'results.csv')):
        inner = os.path.join(d, os.path.basename(d))
        if os.path.exists(os.path.join(inner, 'results.csv')):
            for item in os.listdir(inner):
                shutil.move(os.path.join(inner, item), os.path.join(d, item))
            os.rmdir(inner)

    ok = os.path.exists(os.path.join(d, 'results.csv'))
    print(f'    -> {"berhasil" if ok else "GAGAL: results.csv tidak ada di dalam zip"}')
    return ok


def _inventory():
    """Daftar direktori hasil dan arsip yang benar-benar ada, untuk pesan galat."""
    dirs = sorted(p for p in glob.glob('*')
                  if os.path.isdir(p)
                  and os.path.exists(os.path.join(p, 'results.csv')))
    zips = sorted(os.path.basename(p) for base in ZIP_SEARCH_PATHS
                  for p in glob.glob(os.path.join(base, 'results*.zip')))
    lines = ['\nYang tersedia di direktori kerja saat ini:']
    lines.append('  direktori hasil : ' + (', '.join(dirs) if dirs else '(tidak ada)'))
    lines.append('  arsip zip       : ' + (', '.join(sorted(set(zips))) if zips
                                           else '(tidak ada)'))
    lines.append('')
    lines.append('Jika ini sesi Colab yang baru, /content sudah dikosongkan.')
    lines.append('Unggah kembali zip hasil sesi sebelumnya lewat panel Files')
    lines.append('(ikon folder di kiri), lalu jalankan ulang sel ini — skrip')
    lines.append('akan mengekstraknya sendiri.')
    return '\n'.join(lines)


def validate(dirs):
    """Pastikan SEMUA direktori masukan ada sebelum apa pun ditulis."""
    missing = []
    for d in dirs:
        if not restore_if_needed(d):
            missing.append(d)
        elif not os.path.exists(os.path.join(d, 'results.csv')):
            missing.append(d)
    if missing:
        raise FileNotFoundError(
            'Direktori hasil berikut tidak ditemukan dan tidak ada arsipnya: '
            + ', '.join(missing) + '\n' + _inventory())


def merge(out, dirs):
    # Validasi dijalankan LEBIH DULU. Versi sebelumnya membuat direktori
    # keluaran sebelum memeriksa masukan, sehingga saat merge gagal ia
    # meninggalkan results_uci_all/ yang kosong — dan perintah analyze
    # berikutnya melaporkan galat kedua yang menyesatkan.
    validate(dirs)

    os.makedirs(os.path.join(out, 'proba'), exist_ok=True)
    os.makedirs(os.path.join(out, 'guide'), exist_ok=True)

    res, diag = [], []
    for d in dirs:
        r = pd.read_csv(os.path.join(d, 'results.csv'))
        res.append(r)
        print(f'  {d:24s} {len(r):6d} baris | seed {sorted(r.Seed.unique())}')

        dp = os.path.join(d, 'diagnostics.csv')
        if os.path.exists(dp):
            diag.append(pd.read_csv(dp))

        for sub in ('proba', 'guide'):
            src = os.path.join(d, sub)
            if not os.path.isdir(src):
                continue
            for f in os.listdir(src):
                dst = os.path.join(out, sub, f)
                if os.path.exists(dst):
                    print(f'    ! {sub}/{f} sudah ada — dilewati '
                          f'(seed yang sama dijalankan dua kali?)')
                    continue
                shutil.copy2(os.path.join(src, f), dst)

    R = pd.concat(res, ignore_index=True)
    dup = R.duplicated(subset=['Seed', 'Fold', 'Oversampling', 'Classifier'])
    if dup.any():
        print(f'\n! {int(dup.sum())} baris duplikat (seed x fold x metode x '
              f'classifier) dibuang')
        R = R[~dup]
    R.to_csv(os.path.join(out, 'results.csv'), index=False)

    if diag:
        D = pd.concat(diag, ignore_index=True)
        D = D.drop_duplicates(subset=['Seed', 'Fold', 'Method', 'Class'])
        D.to_csv(os.path.join(out, 'diagnostics.csv'), index=False)
    else:
        D = pd.DataFrame()

    for d in dirs:
        cp = os.path.join(d, 'config.json')
        if os.path.exists(cp):
            shutil.copy2(cp, os.path.join(out, 'config.json'))
            break

    print(f'\nGabungan -> {out}/')
    print(f'  results.csv     : {len(R)} baris, seed {sorted(R.Seed.unique())}')
    print(f'  diagnostics.csv : {len(D)} baris')
    print(f'  proba/          : {len(os.listdir(os.path.join(out, "proba")))} berkas')
    print(f'  guide/          : {len(os.listdir(os.path.join(out, "guide")))} berkas')

    n_exp = R.Seed.nunique() * R.Fold.nunique() * R.Oversampling.nunique() * \
        R.Classifier.nunique()
    print(f'  kelengkapan     : {len(R)}/{n_exp} '
          f'({100 * len(R) / n_exp:.1f}%)')
    if len(R) < n_exp:
        print('  ! tidak lengkap — periksa apakah ada seed/fold yang gagal')
    return R, D


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('dirs', nargs='+')
    a = ap.parse_args()
    merge(a.out, a.dirs)
