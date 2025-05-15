# ESP Intelligence

Dashboard analisis untuk memantau dan menganalisis data Electrical Submersible Pump (ESP) di industri minyak dan gas.

## Deskripsi

ESP Intelligence adalah aplikasi berbasis Streamlit yang menyediakan analisis dan visualisasi komprehensif untuk data Electrical Submersible Pump (ESP). Aplikasi ini membantu engineer dan manajer di industri minyak dan gas untuk memahami performa ESP, mengidentifikasi penyebab kegagalan, dan mengoptimalkan operasi sumur minyak.

## Fitur Utama

- **Natural Language Query**: Tanyakan tentang data ESP dalam bahasa alami
- **Analisis Vendor**: Bandingkan performa vendor ESP berdasarkan run life, failure rate, dan metrik lainnya
- **Analisis Kegagalan**: Identifikasi penyebab umum kegagalan dan komponen yang sering rusak
- **Analisis Area**: Bandingkan performa ESP berdasarkan lokasi geografis
- **Analisis Run Life**: Lihat distribusi dan trend run life seiring waktu
- **Visualisasi Interaktif**: Berbagai jenis grafik untuk eksplorasi data
- **Analisis AI**: Dapatkan insight dan rekomendasi berdasarkan data

## Dataset

Aplikasi ini menggunakan tiga dataset utama:
- `wells_cleaned.csv`: Informasi dasar tentang sumur minyak
- `esp_well_installations_cleaned.csv`: Data instalasi ESP di sumur
- `esp_failure_cause_cleaned.csv`: Data kegagalan ESP dan penyebabnya

## Instalasi

1. Clone repository ini:
   ```bash
   git clone https://github.com/caernations/esp-intelligence
   cd esp-intelligence
   ```

2. Buat dan aktifkan virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Untuk Linux/Mac
   # atau
   .venv\Scripts\activate  # Untuk Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Buat file `.env` untuk menyimpan API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Menjalankan Aplikasi

Jalankan aplikasi Streamlit dengan perintah:

```bash
streamlit run main.py
```

Aplikasi akan terbuka di browser dengan alamat http://localhost:8501

## Cara Penggunaan

1. **Natural Language Query**:
   - Masukkan pertanyaan tentang data ESP dalam bahasa alami
   - Contoh: "Vendor mana dengan performa terbaik?" atau "Apa penyebab utama kegagalan ESP?"

2. **Guided Analysis**:
   - Pilih jenis analisis dari dropdown
   - Atur parameter untuk menyesuaikan analisis

3. **Data Exploration**:
   - Lihat visualisasi dalam tab "Visualisasi"
   - Eksplorasi insight tambahan dalam tab "Insights"
   - Lihat dan filter data mentah dalam tab "Data Explorer"
   - Dapatkan analisis otomatis dalam tab "AI Analysis"

## Struktur Kode

- `main.py`: File utama aplikasi Streamlit
- `data/`: Direktori berisi dataset CSV
- `.venv/`: Virtual environment Python
- `requirements.txt`: Daftar package Python yang dibutuhkan

## Teknologi

- **Streamlit**: Framework untuk aplikasi data science
- **Pandas**: Manipulasi dan analisis data
- **Plotly**: Visualisasi data interaktif
- **OpenAI API**: Untuk fitur analisis berbasis AI
- **Matplotlib & Seaborn**: Visualisasi data tambahan

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- NumPy
- OpenAI
- Python-dotenv