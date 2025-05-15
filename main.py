import os
import pandas as pd
import streamlit as st
import openai
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt

# Set page config HARUS DI PANGGIL PERTAMA KALI
st.set_page_config(page_title="ESP Data Q&A & Visualization", layout="wide")

# Load env & API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Load CSV data & cache
@st.cache_data
def load_data():
    wells = pd.read_csv("data/wells_cleaned.csv")
    installations = pd.read_csv("data/esp_well_installations_cleaned.csv")
    failure_cause = pd.read_csv("data/esp_failure_cause_cleaned.csv")
    return wells, installations, failure_cause

wells_df, installations_df, failure_cause_df = load_data()

# Fungsi interpretasi query user jadi keyword/trigger
def interpret_query(query):
    q = query.lower()
    if any(word in q for word in ["vendor", "pemasok", "penyedia", "supplier", "produsen", "buat", "pabrikan", "manufacturer", "maker", "vendor terbaik", "vendor mana"]):
        return "vendor"
    if any(word in q for word in ["failure", "gagal", "kerusakan", "rusak", "mode failure", "penyebab gagal", "kegagalan", "failure mode", "macet", "dana", "loss"]):
        return "failure_mode"
    if any(word in q for word in ["area", "wilayah", "cluster", "zona", "daerah", "lokasi", "tempat"]):
        return "area"
    if any(word in q for word in ["pump", "pompa", "tipe pompa", "pump type", "jenis pompa"]):
        return "pump_type"
    if any(word in q for word in ["status", "keadaan", "aktif", "mati", "produksi"]):
        return "status"
    if any(word in q for word in ["run life", "umur pakai", "run", "durasi", "waktu operasional", "waktu berjalan"]):
        return "runlife"
    if any(word in q for word in ["rl code", "kategori", "warna", "performance category"]):
        return "rl_code"
    if any(word in q for word in ["failure item", "komponen gagal", "bagian rusak", "komponen", "item gagal"]):
        return "failure_item"
    return "wells"

# Fungsi filter dan agregasi data sesuai keyword
def filter_data(keyword):
    if keyword == "vendor":
        df = installations_df.groupby("vendor").agg({'run': 'mean', 'trl': 'mean'}).reset_index()
        return df, "vendor", "run"
    elif keyword == "failure_mode":
        df = failure_cause_df.groupby("failure_mode").size().reset_index(name='count')
        return df, "failure_mode", "count"
    elif keyword == "area":
        df = installations_df.groupby("area").agg({'run': 'mean'}).reset_index()
        return df, "area", "run"
    elif keyword == "pump_type":
        df = installations_df.groupby("pump_type").agg({'run': 'mean'}).reset_index()
        return df, "pump_type", "run"
    elif keyword == "status":
        df = installations_df.groupby("status").size().reset_index(name='count')
        return df, "status", "count"
    elif keyword == "runlife":
        df = installations_df.groupby("rl_code").agg({'run': 'mean'}).reset_index()
        return df, "rl_code", "run"
    elif keyword == "rl_code":
        df = installations_df.groupby("rl_code").size().reset_index(name='count')
        return df, "rl_code", "count"
    elif keyword == "failure_item":
        df = failure_cause_df.groupby("failure_item").size().reset_index(name='count')
        return df, "failure_item", "count"
    else:
        df = wells_df.head(10)
        return df, None, None

# Fungsi plot dinamis
def plot_data(df, plot_type, x_col, y_col):
    fig, ax = plt.subplots(figsize=(9,5))
    if plot_type == "bar":
        ax.bar(df[x_col], df[y_col], color="mediumslateblue")
        ax.set_xlabel(x_col.title())
        ax.set_ylabel(y_col.title())
        ax.set_title(f"{y_col.title()} per {x_col.title()}")
        plt.xticks(rotation=45, ha='right')
    elif plot_type == "line":
        ax.plot(df[x_col], df[y_col], marker='o', color="mediumseagreen")
        ax.set_xlabel(x_col.title())
        ax.set_ylabel(y_col.title())
        ax.set_title(f"{y_col.title()} per {x_col.title()} (Line Chart)")
        plt.xticks(rotation=45, ha='right')
    elif plot_type == "pie":
        ax.pie(df[y_col], labels=df[x_col], autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)
        ax.set_title(f"Distribusi {y_col.title()} per {x_col.title()}")
        ax.axis('equal')
    else:
        return None
    plt.tight_layout()
    return fig

# Deskripsi data yang kamu kasih, ditampilkan collapsible agar rapi
with st.expander("📊 Tentang Data ESP yang digunakan (klik untuk baca)"):
    st.markdown("""
    1. Wells.csv - File Master Sumur
File ini merupakan file master yang berisi daftar lengkap semua sumur yang dikelola. Berikut fitur-fiturnya:
Fitur:
Well Name: Nama unik setiap sumur dengan format standar (contoh: AA-01, CB-05, WB-12)
Format penamaan mengikuti pola: [Area Code]-[Nomor Sumur][Suffix opsional]
Suffix yang mungkin ada:
S: Sidetrack (sumur cabang)
ST: Sidetrack
i: Injection well (sumur injeksi)
CAN A/B: Sumur cadangan (Candidate wells)
File ini berisi 894 sumur dari berbagai area operasi, berfungsi sebagai referensi utama untuk menghubungkan data antar file.
2. ESP Well Installations.csv - File Riwayat Instalasi ESP
File ini mencatat setiap instalasi ESP pada sumur. Satu sumur bisa memiliki multiple records karena pompa ESP diganti berkali-kali selama masa operasi. Berikut penjelasan lengkap fitur-fiturnya:
Fitur Identifikasi:
Well: Nama sumur (foreign key ke wells.csv)
Area: Area/cluster geografis sumur (AA, AB, BA, dll.)
Fitur Instalasi:
PSD (Pump Setting Depth): Kedalaman pemasangan pompa dalam sumur (meter)
Start Date: Tanggal instalasi ESP dimulai
DHP Date: Tanggal terjadinya Downhole Problem (kerusakan permanen)
Pulling Date: Tanggal ESP ditarik keluar dari sumur (jika ada)
Spesifikasi Teknis ESP:
PSD (Pump Setting Depth): Kedalaman pemasangan pompa ESP di dalam sumur (meter/feet)
Pump Type: Tipe/model pompa yang digunakan
STG: Jumlah stage/tingkat pompa
HP: Horsepower/daya kuda motor
Volt: Voltase operasional
Amp: Ampere/arus listrik
BOI/GS: Bottom Oil Intake/Gas Separator - komponen untuk memisahkan gas
AGH/PGP: Auto Gas Handler/Pump Gas Protector
Shroud: Pelindung pompa
Protector: Pelindung motor dari fluida sumur
Sensor: Sensor untuk monitoring kondisi ESP
Fitur Vendor & Kontrak:
Vendor: Pembuat ESP (Baker, Reda, Powerlift, Other)
Current Pack: Jenis paket kontrak/pembelian
Purchase/Rent: Model bisnis (Purchase, Rental, Own, Repair)
Fitur Teknis ESP:
Pump Type: Tipe/model pompa
STG: Jumlah stage pompa (tingkat pemompaan)
HP: Horsepower/daya motor
Volt: Voltase operasional
Amp: Arus listrik nominal
BOI/GS: Bottom Intake/Gas Separator
AGH/PGP: Advanced Gas Handler/Poor Gas Performance
Shroud: Pelindung pompa
Protector: Unit pelindung motor
Sensor: Jenis sensor yang terpasang
Fitur Performance & Analisis:
TRL (Target Run Life): Target umur pakai dalam hari (berdasarkan kontrak/historis)
RUN: Actual run life (jika sudah DHP) atau current runtime (jika masih aktif)
STATUS: P (Producing/aktif) atau DHP (sudah rusak)
Fitur Kategorisasi Performance:
RL Weight: Skor numerik (1-4) untuk kategorisasi durasi
4: >465 hari (Excellent)
3: 366-465 hari (Good)
2: 71-365 hari (Fair)
1: ≤70 hari (Poor)
RL Code: Kode warna performance
GREEN: >465 hari
YELLOW: 366-465 hari
ORANGE: 71-365 hari
RED: ≤70 hari
Fitur Mode Kegagalan:
Failure Mode: Jenis kegagalan
MDHP: Mechanical Downhole Problem
EDHP: Electrical Downhole Problem
DHP TL: Downhole Problem Tubing Leak
MOFF: Manual Off (dimatikan operator)
Others: Lain-lain
Fitur Historis:
lastDHP, lastDHP2, lastDHP3: Tanggal DHP 1-3 instalasi sebelumnya
RUN1, RUN2, RUN3: Run life instalasi sebelumnya
ARL3Last: Average Run Life 3 instalasi terakhir
3. ESP Failure Cause.csv - File Analisis Kegagalan
File ini berisi hasil analisis mendalam setiap kegagalan ESP untuk memahami root cause dan mencegah kegagalan serupa. Berikut fitur lengkapnya:
Fitur Identifikasi:
Well: Nama sumur
Area: Area/cluster sumur
Manufacture: Vendor pembuat ESP (sama dengan kolom Vendor di installations)
Fitur Timeline:
Installation Date: Tanggal pasang
DHP Date: Tanggal kegagalan
Pulling Date: Tanggal pompa ditarik dari sumur
DIFA Date: Tanggal Dismantle Inspection Failure Analysis
Fitur Performance:
Run: Durasi operasi dalam hari
TRL: Target Run Life
RL Code: Kategorisasi performance (sama dengan installations)
Current Pack: Tipe paket kontrak
Klasifikasi Kegagalan:
Failure Mode: Jenis kegagalan (MDHP, EDHP, DHP TL, MOFF, Others)
Failure Item: Komponen utama yang gagal (motor, protector, pump, cable)
Failure Item Specific: Sub-komponen spesifik yang gagal
Fitur Klasifikasi Kegagalan:
General Failure Descriptor: Deskripsi umum kegagalan
Detailed Failure Descriptor: Deskripsi detail kegagalan
General Failure Cause: Penyebab umum kegagalan
Specific Failure Cause: Penyebab spesifik kegagalan
Fitur Analisis Komponen:
Failure Item: Komponen utama yang gagal (motor, pump, protector, cable, dll.)
Failure Item Specific: Sub-komponen spesifik yang gagal
Fitur Detail per Komponen:
Failure Component Pump: Detail kegagalan pompa
Failure Component Motor: Detail kegagalan motor
Failure Component Protector: Detail kegagalan protector
Failure Component MLE: Detail kegagalan Motor Lead Extension
Failure Component Main Cable: Detail kegagalan kabel utama
Failure Component Gas Separator/Intake: Detail kegagalan separator gas
Fitur Rekomendasi:
Recommendation: Saran teknis untuk pencegahan kegagalan serupa
Fitur Tahapan Investigasi:
Data Reference: Tahap investigasi (Preliminary, Pulling, DIFA)
Preliminary: Analisis awal dari data operasional
Pulling: Inspeksi visual setelah ditarik
DIFA: Pembongkaran dan analisis komprehensif
Tujuan Integrasi Ketiga File:
Wells.csv: Database master untuk identifikasi dan referensi sumur
ESP Well Installations.csv: Tracking operasional, performance monitoring, dan trend analysis
ESP Failure Cause.csv: Root cause analysis, vendor evaluation, dan continuous improvement
Ketiga file ini membentuk sistem data lengkap untuk:
Monitoring kesehatan aset ESP
Evaluasi vendor dan teknologi
Prediksi kegagalan dan maintenance planning
Cost optimization dan contract management
Technical improvement dan best practices



    ---  
    Ketiga file ini membentuk sistem data lengkap untuk monitoring aset ESP, evaluasi vendor, prediksi kegagalan, dan optimasi biaya.
    """)

# UI Streamlit Q&A & Visualisasi
st.title("🔥 ESP Data Q&A with Dynamic Visualizations")

user_query = st.text_input("Tanya apa saja tentang data ESP:")

if user_query:
    st.info("Memproses data dan menyiapkan visualisasi...")
    keyword = interpret_query(user_query)
    df, x_col, y_col = filter_data(keyword)

    if x_col and y_col:
        chart_type = st.selectbox("Pilih tipe grafik:", ["bar", "line", "pie"])
        fig = plot_data(df, chart_type, x_col, y_col)
        if fig:
            st.pyplot(fig)
        st.write(df)
    else:
        st.write(df)

    # Prompt ke LLM
    filtered_text = df.to_string(index=False)
    prompt = f"Berikut data ESP terkait:\n{filtered_text}\n\nJawab pertanyaan ini:\n{user_query}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Kamu asisten yang membantu menjawab berdasarkan data."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.3,
    )
    answer = response.choices[0].message.content
    st.markdown(f"### Jawaban:\n{answer}")
else:
    st.write("Masukkan pertanyaan kamu di atas untuk mulai.")
