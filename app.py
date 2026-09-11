import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page config for premium look
st.set_page_config(
    page_title="Tomato Leaf Disease Detector",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI (Red & Green Theme)
st.markdown("""
<style>
    .main {
        background-color: #f4fcf4; /* Very light green background */
    }
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    h1 {
        color: #2e7d32; /* Dark green for title */
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .result-card {
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        animation: fadeIn 0.8s ease-out;
        border: 2px solid white;
    }
    .result-card-healthy {
        background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%); /* Solid Green */
    }
    .result-card-disease {
        background: linear-gradient(135deg, #c62828 0%, #f44336 100%); /* Solid Red */
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }
</style>
""", unsafe_allow_html=True)

# Define generic classes (Assuming alphabetical PlantVillage order)
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

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Muhamad Rizal Fikri_2612_Week2BD_LAS26_Soal2.h5")
    return model

def predict_image(image, model):
    # Resize to match model optimal resolution
    image = image.resize((150, 150))
    # Convert image to array
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    # Expand dimensions to create a batch (1, 150, 150, 3)
    img_array = tf.expand_dims(img_array, 0)
    
    # The model already contains a Rescaling layer internally!
    # Therefore, we MUST NOT divide by 255.0 manually, otherwise it will be scaled twice.
    # img_array = img_array / 255.0
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    return CLASS_NAMES[predicted_class_idx], confidence

def main():
    st.title("🍅 Tomato Leaf Disease Detection")
    
    st.sidebar.title("Informasi Sistem")
    st.sidebar.info(
        "Aplikasi ini menggunakan model Custom Convolutional Neural Network (CNN) "
        "untuk mendiagnosis kondisi kesehatan daun tomat berdasarkan tekstur dan visualisasi."
    )
    st.sidebar.markdown("### Spesifikasi")
    st.sidebar.write("- **Arsitektur:** Custom CNN (Deeper Network)")
    st.sidebar.write("- **Input:** Gambar RGB 150x150")
    st.sidebar.write("- **Classes:** 10 Kategori Penyakit & Sehat")
    
    st.write("Silakan unggah gambar daun tomat untuk dianalisis secara presisi oleh sistem AI kami.")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.write("### 📤 Upload Daun")
        uploaded_file = st.file_uploader("Pilih gambar daun tomat (JPG/PNG)...", type=["jpg", "jpeg", "png"])
        
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            with col1:
                st.image(image, caption='Pratinjau Gambar', use_container_width=True)
            
            with col2:
                st.write("### 🤖 Analisis Sistem")
                with st.spinner("Menjalankan diagnosis mikroskopis..."):
                    model = load_model()
                    label, conf = predict_image(image, model)
                    
                    # Display Results
                    st.write("#### Hasil Diagnosis:")
                    is_healthy = (label == "Healthy")
                    card_class = "result-card-healthy" if is_healthy else "result-card-disease"
                    
                    st.markdown(f"""
                    <div class="result-card {card_class}">
                        <h2 style="color: white; margin-bottom: 0;">{label}</h2>
                        <p style="font-size: 1.2rem; margin-top: 5px;">Akurasi / Keyakinan: <strong>{conf*100:.2f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    if not is_healthy:
                        st.warning("⚠️ **Peringatan:** Terdeteksi indikasi penyakit pada tanaman. Lakukan isolasi tanaman atau semprotkan fungisida/bakterisida sesuai jenis patogen untuk mencegah gagal panen.")
                    else:
                        st.success("✅ **Aman:** Tidak ditemukan indikasi patogen berbahaya. Lanjutkan perawatan nutrisi secara rutin.")
                        
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

if __name__ == '__main__':
    main()
