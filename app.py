import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd

# -------------------------------------------------------------
# 1. Konfigurasi Halaman & Styling Modern
# -------------------------------------------------------------
st.set_page_config(
    page_title="AgriVision Tomato AI - Smart Farming Assistant",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Theme: High Contrast, Clean & Compatible with Dark/Light Mode
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #092f13 0%, #14532d 50%, #166534 100%);
        color: #ffffff !important;
        padding: 26px 30px;
        border-radius: 18px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 8px;
        color: #ffffff !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #dcfce7 !important;
        margin-bottom: 0;
        line-height: 1.5;
    }

    .badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 700;
        background: rgba(255, 255, 255, 0.22);
        color: #ffffff !important;
        margin-right: 8px;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }
    
    .result-box {
        border-radius: 16px;
        padding: 24px;
        color: #ffffff !important;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
        animation: fadeIn 0.5s ease-out;
    }
    
    .result-healthy {
        background: linear-gradient(135deg, #14532d 0%, #15803d 100%);
        border: 2px solid #86efac;
    }
    
    .result-disease {
        background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%);
        border: 2px solid #fca5a5;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Database Pengetahuan Penyakit Daun Tomat
# -------------------------------------------------------------
CLASS_NAMES = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites (Two-spotted spider mite)",
    "Target Spot",
    "Yellow Leaf Curl Virus",
    "Mosaic Virus",
    "Healthy"
]

DISEASE_DETAILS = {
    "Bacterial Spot": {
        "nama_ilmiah": "Xanthomonas campestris pv. vesicatoria",
        "tipe": "Bakteri",
        "gejala": "Bercak kecil basah berwarna coklat tua hingga hitam dengan tepi bergerigi pada daun dan buah.",
        "solusi": [
            "Semprotkan bakterisida berbasis tembaga (copper bactericide) saat fase vegetatif.",
            "Hindari penyiraman daun dari atas (gunakan sistem irigasi tetes/pangkal batang).",
            "Segera pangkas daun bawah yang terinfeksi parah dan musnahkan (jangan dikompos)."
        ],
        "pencegahan": "Gunakan benih bersertifikat bebas penyakit dan lakukan rotasi tanaman minimal 2 tahun."
    },
    "Early Blight": {
        "nama_ilmiah": "Alternaria solani",
        "tipe": "Jamur / Fungi",
        "gejala": "Bercak coklat gelap melingkar konsentris menyerupai papan sasaran (target board pattern), biasanya dimulai dari daun tertua.",
        "solusi": [
            "Aplikasikan fungisida berbahan aktif Mankozeb, Klorotalonil, atau Azoksistrobin.",
            "Pangkas daun tua paling bawah yang menempel ke permukaan tanah.",
            "Jaga sirkulasi udara dengan menata jarak tanam yang tidak terlalu rapat."
        ],
        "pencegahan": "Beri mulsa plastik untuk mencegah percikan spora jamur dari tanah ke daun saat hujan."
    },
    "Late Blight": {
        "nama_ilmiah": "Phytophthora infestans",
        "tipe": "Oomycota (Jamur Air Ganas)",
        "gejala": "Bercak basah abu-abu kehijauan cepat membesar menjadi coklat berminyak, lapisan beludru putih di balik daun saat cuaca lembap.",
        "solusi": [
            "Terapkan fungisida sistemik spektrum luas (seperti Simoksanil atau Dimetomorf) secepatnya.",
            "Cabut tanaman yang terinfeksi akut untuk memotong rantai infeksi ke kebun sekitar.",
            "Jangan biarkan daun basah berlama-lama pada malam hari."
        ],
        "pencegahan": "Monitoring ekstra saat musim hujan atau kondisi berkabut dingin dengan kelembapan tinggi."
    },
    "Leaf Mold": {
        "nama_ilmiah": "Passalora fulva (Cladosporium fulvum)",
        "tipe": "Jamur / Fungi",
        "gejala": "Bercak kuning pucat di permukaan atas daun, disertai massa spora beludru berwarna zaitun kecoklatan di permukaan bawah.",
        "solusi": [
            "Tingkatkan ventilasi dan aerasi greenhouse untuk menurunkan kelembapan relatif (<80%).",
            "Semprotkan fungisida protektan berbasis tembaga atau Difenokonazol.",
            "Pangkas daun rimbun yang menghalangi sirkulasi udara."
        ],
        "pencegahan": "Gunakan varietas tahan jamur daun dan jaga suhu ruang tanam agar stabil."
    },
    "Septoria Leaf Spot": {
        "nama_ilmiah": "Septoria lycopersici",
        "tipe": "Jamur / Fungi",
        "gejala": "Bintik-bintik bulat kecil berdiameter 1-3 mm dengan bagian tengah abu-abu dan pinggiran coklat kehitaman.",
        "solusi": [
            "Gunakan fungisida Klorotalonil atau tembaga hidroksida.",
            "Bersihkan gulma dan sisa tanaman mati di sekitar bedengan.",
            "Sterilkan gunting pangkas sebelum berpindah ke tanaman lain."
        ],
        "pencegahan": "Rotasi tanaman dengan famili non-Solanaceae (hindari menanam setelah terong/cabai)."
    },
    "Spider Mites (Two-spotted spider mite)": {
        "nama_ilmiah": "Tetranychus urticae",
        "tipe": "Hama Tungau",
        "gejala": "Bintik-bintik klorosis kuning halus (stippling) di permukaan daun, disusul anyaman jaring laba-laba halus di bawah daun.",
        "solusi": [
            "Semprotkan akarisida alami (Minyak Nimba / Neem Oil) atau insektisida Abamektin.",
            "Semprotkan air bertekanan ke bagian bawah daun untuk merontokkan koloni tungau.",
            "Manfaatkan musuh alami seperti predator tungau (Phytoseiidae)."
        ],
        "pencegahan": "Hindari kekeringan tanah dan debu berlebih karena tungau berkembang biak pesat di iklim panas kering."
    },
    "Target Spot": {
        "nama_ilmiah": "Corynespora cassiicola",
        "tipe": "Jamur / Fungi",
        "gejala": "Bercak nekrotik coklat dengan cincin konsentris berbatas tegas pada daun dan batang.",
        "solusi": [
            "Semprotkan fungisida Azoksistrobin atau Fludioksonil.",
            "Kurangi kepadatan kanopi tanaman melalui pemangkasan tunas air.",
            "Pastikan drainase tanah baik dan tidak ada air menggenang."
        ],
        "pencegahan": "Hindari menanam tomat berdekatan dengan tanaman inang lain seperti kedelai atau mentimun."
    },
    "Yellow Leaf Curl Virus": {
        "nama_ilmiah": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "tipe": "Virus (Vektor Kutu Kebul / Whitefly)",
        "gejala": "Daun menggulung ke atas seperti mangkuk, tepi daun menguning (klorotik), tanaman kerdil dan bunga gugur.",
        "solusi": [
            "Kendalikan hama vektor kutu kebul (*Bemisia tabaci*) dengan perangkap kuning (Yellow Sticky Trap).",
            "Semprot insektisida Imidakloprid atau Sabun Kalium Organik.",
            "Cabut tanaman yang telah tertular parah untuk mencegah penularan ke tanaman sehat."
        ],
        "pencegahan": "Gunakan bibit bersertifikat toleran TYLCV dan pasang jaring serangga (insect screen) pada greenhouse."
    },
    "Mosaic Virus": {
        "nama_ilmiah": "Tomato Mosaic Virus (ToMV)",
        "tipe": "Virus Mekanis",
        "gejala": "Pola belang-belang mosaik hijau tua dan hijau muda, permukaan daun mengkerut/melepuh, distorsi seperti tali sepatu.",
        "solusi": [
            "Tidak ada obat kimia untuk virus tanaman yang sudah masuk ke jaringan vaskular.",
            "Segera cabut dan bakar tanaman terinfeksi.",
            "Sterilkan tangan dan peralatan kerja dengan larutan susu skim atau deterjen sebelum menyentuh tanaman lain."
        ],
        "pencegahan": "Dilarang merokok di area kebun (tembakau membawa TMV) dan gunakan benih bebas virus."
    },
    "Healthy": {
        "nama_ilmiah": "Solanum lycopersicum",
        "tipe": "Tanaman Sehat",
        "gejala": "Permukaan daun hijau segar merata, turgor baik, tidak ada lesi bercak, nekrosis, maupun infestasi hama.",
        "solusi": [
            "Pertahankan jadwal pemupukan berimbang (NPK + unsur mikro Ca, Mg, B).",
            "Penyiraman rutin secukupnya di pagi hari pada area perakaran.",
            "Lakukan pemantauan mingguan secara konsisten untuk deteksi dini."
        ],
        "pencegahan": "Terapkan Good Agricultural Practices (GAP) secara konsisten."
    }
}

# -------------------------------------------------------------
# 3. Cache Model Loader & Prediksi
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Muhamad Rizal Fikri_2612_Week2BD_LAS26_Soal2.h5")

def predict_image(image, model):
    # Resize 150x150
    image_resized = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    img_array = tf.expand_dims(img_array, 0)
    
    # Model memiliki layer Rescaling bawaan, jangan dibagi 255 manual
    preds = model.predict(img_array)[0]
    
    # Urutkan dari confidence tertinggi
    top_indices = np.argsort(preds)[::-1]
    top_results = [(CLASS_NAMES[i], float(preds[i])) for i in top_indices]
    
    return top_results

# -------------------------------------------------------------
# 4. Antarmuka Utama (UI Layout)
# -------------------------------------------------------------
def main():
    # Hero Header Banner
    st.markdown("""
    <div class="hero-container">
        <span class="badge">🌱 Precision Agriculture</span>
        <span class="badge">🧠 Custom CNN Deeper Network</span>
        <span class="badge">⚡ Realtime Diagnosis</span>
        <h1 class="hero-title">🍅 AgriVision: Tomato Leaf Doctor</h1>
        <p class="hero-subtitle">
            Sistem cerdas klasifikasi penyakit tanaman tomat 10 kelas berbasis visual daun mikroskopis.
            Membantu petani mendeteksi patogen secara cepat dan akurat untuk mencegah gagal panen.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Informasi & Panduan
    st.sidebar.markdown("## 🌿 Panel Kontrol & Info")
    st.sidebar.info(
        "Aplikasi ini digerakkan oleh arsitektur **Custom CNN** dengan 4 lapisan konvolusi bertingkat "
        "(32, 64, 128, 256 filter) yang dioptimasi khusus untuk tekstur dan klorosis daun tomat."
    )
    
    with st.sidebar.expander("📊 Spesifikasi Model", expanded=True):
        st.write("- **Model:** Custom CNN Scratch")
        st.write("- **Resolusi:** 150 x 150 piksel (RGB)")
        st.write("- **Total Kelas:** 10 Kategori")
        st.write("- **Deployment:** Streamlit Cloud Ready")

    with st.sidebar.expander("💡 Tips Pengambilan Gambar"):
        st.markdown("""
        1. **Fokus pada Daun:** Pastikan daun tampak jelas memenuhi frame.
        2. **Pencahayaan Baik:** Hindari bayangan gelap berlebih.
        3. **Latar Belakang Netral:** Latar polos memudahkan ekstraksi tekstur bercak.
        """)

    # Main Tabs
    tab_scan, tab_catalog, tab_history = st.tabs([
        "🔬 Diagnosis Daun", 
        "📚 Ensiklopedia Penyakit", 
        "📋 Riwayat Sesi"
    ])
    
    # Inisialisasi riwayat sesi
    if "scan_history" not in st.session_state:
        st.session_state.scan_history = []
        
    # ---------------------------------------------------------
    # TAB 1: DIAGNOSIS DAUN
    # ---------------------------------------------------------
    with tab_scan:
        col_input, col_result = st.columns([1, 1.2], gap="large")
        
        with col_input:
            st.markdown("### 📸 Input Daun Tomat")
            input_mode = st.radio(
                "Pilih Metode Pengambilan:",
                ["📁 Unggah File Gambar", "📷 Ambil dengan Kamera"],
                horizontal=True
            )
            
            image = None
            if input_mode == "📁 Unggah File Gambar":
                uploaded_file = st.file_uploader(
                    "Pilih foto daun tomat (JPG, JPEG, PNG)...", 
                    type=["jpg", "jpeg", "png"]
                )
                if uploaded_file is not None:
                    image = Image.open(uploaded_file).convert("RGB")
            else:
                camera_file = st.camera_input("Arahkan kamera ke daun tomat...")
                if camera_file is not None:
                    image = Image.open(camera_file).convert("RGB")
            
            if image is not None:
                st.image(image, caption="Gambar yang siap dianalisis", use_container_width=True)
                
                # Preview thumbnail 150x150
                with st.expander("🔍 Lihat Input Matriks Model (150x150)"):
                    st.image(image.resize((150, 150)), width=150, caption="Normalized Matrix Feed")
        
        with col_result:
            st.markdown("### 🩺 Hasil Diagnosis AI")
            
            if image is None:
                st.info("👈 **Belum ada daun yang dianalisis.**\n\nSilakan unggah foto daun tomat atau potret menggunakan kamera di panel kiri untuk melihat hasil diagnosis otomatis.")
            else:
                with st.spinner("🔬 Menganalisis pola bercak, klorosis, dan tekstur daun..."):
                    model = load_model()
                    top_results = predict_image(image, model)
                    
                    best_label, best_conf = top_results[0]
                    is_healthy = (best_label == "Healthy")
                    
                    # Simpan ke riwayat sesi jika belum disimpan
                    if not any(item["label"] == best_label and item["conf"] == best_conf for item in st.session_state.scan_history):
                        st.session_state.scan_history.insert(0, {
                            "label": best_label,
                            "conf": best_conf,
                            "is_healthy": is_healthy
                        })
                    
                    # Kartu Hasil Utama
                    box_class = "result-healthy" if is_healthy else "result-disease"
                    status_icon = "✅ STATUS: SEHAT" if is_healthy else "⚠️ STATUS: TERDETEKSI PENYAKIT"
                    
                    st.markdown(f"""
                    <div class="result-box {box_class}">
                        <div style="font-size: 0.95rem; letter-spacing: 1px; font-weight: 800; color: #ffffff !important;">{status_icon}</div>
                        <h2 style="color: #ffffff !important; margin: 8px 0 4px 0; font-size: 2.2rem; font-weight: 800;">{best_label}</h2>
                        <p style="margin: 0; font-size: 1.2rem; font-weight: 600; color: #ffffff !important;">
                            Tingkat Keyakinan: {best_conf*100:.2f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detail Penyakit & Rekomendasi Penanganan
                    detail = DISEASE_DETAILS.get(best_label, {})
                    
                    st.markdown("#### 📖 Informasi Patogen & Karakteristik")
                    st.markdown(f"- **Penyebab:** *{detail.get('nama_ilmiah', '-')}* (`{detail.get('tipe', '-')}`)")
                    st.markdown(f"- **Gejala Khas:** {detail.get('gejala', '-')}")
                    
                    if is_healthy:
                        st.success("🌱 **Kondisi Prima:** Daun menunjukkan integritas sel yang sehat tanpa indikasi patogen berbahaya. Lanjutkan sanitasi rutin dan pemupukan N-P-K berimbang.")
                    else:
                        st.error("⚡ **Tindakan Diperlukan:** Terdeteksi indikasi patogen aktif. Lakukan tindakan kuratif berikut untuk memotong transmisi penyakit ke tanaman sekitar.")
                        
                    with st.expander("🛠️ Langkah Rekomendasi Penanganan (Treatment)", expanded=True):
                        for step in detail.get("solusi", []):
                            st.write(f"- {step}")
                        st.info(f"**Strategi Preventif:** {detail.get('pencegahan', '-')}")

                    # Top-3 Probabilitas Visual
                    st.markdown("---")
                    st.markdown("#### 📊 Distribusi Probabilitas (Top 3 Model)")
                    for label, conf in top_results[:3]:
                        col_l, col_b = st.columns([1.5, 2.5])
                        with col_l:
                            st.write(f"**{label}**")
                        with col_b:
                            st.progress(float(conf), text=f"{conf*100:.2f}%")

    # ---------------------------------------------------------
    # TAB 2: ENSIKLOPEDIA PENYAKIT
    # ---------------------------------------------------------
    with tab_catalog:
        st.markdown("### 📚 Ensiklopedia Penyakit Daun Tomat")
        st.write("Katalog lengkap 10 kelas kondisi daun tomat yang dapat dideteksi oleh model ini:")
        
        search_query = st.text_input("🔍 Cari penyakit atau patogen...", "")
        
        for name, data in DISEASE_DETAILS.items():
            if search_query.lower() in name.lower() or search_query.lower() in data["nama_ilmiah"].lower() or search_query.lower() in data["tipe"].lower():
                badge_color = "#2e7d32" if name == "Healthy" else "#c62828"
                with st.expander(f"📌 {name} ({data['tipe']})"):
                    st.markdown(f"**Nama Ilmiah:** *{data['nama_ilmiah']}*")
                    st.markdown(f"**Gejala Lapangan:** {data['gejala']}")
                    st.markdown("**Solusi Penanganan:**")
                    for s in data["solusi"]:
                        st.write(f"- {s}")
                    st.markdown(f"**Pencegahan Jangka Panjang:** {data['pencegahan']}")

    # ---------------------------------------------------------
    # TAB 3: RIWAYAT SESI
    # ---------------------------------------------------------
    with tab_history:
        st.markdown("### 📋 Riwayat Diagnosis Sesi Ini")
        if not st.session_state.scan_history:
            st.info("Belum ada riwayat pemindaian pada sesi ini. Unggah gambar di Tab 'Diagnosis Daun'.")
        else:
            df_history = pd.DataFrame(st.session_state.scan_history)
            df_history["conf"] = df_history["conf"].apply(lambda x: f"{x*100:.2f}%")
            df_history.columns = ["Hasil Diagnosis", "Akurasi Keyakinan", "Status Sehat"]
            st.dataframe(df_history, use_container_width=True)
            
            if st.button("🗑️ Bersihkan Riwayat"):
                st.session_state.scan_history = []
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 0.85rem; padding-bottom: 20px;'>"
        "AgriVision AI © 2026 • Powered by TensorFlow & Streamlit • Custom CNN Agriculture Model"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
