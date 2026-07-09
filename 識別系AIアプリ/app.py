import os
import streamlit as st
import numpy as np
from PIL import Image

# ==========================================
# 📂 フォルダ・パス設定
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
ILLUST_DIR = os.path.join(BASE_DIR, "illust")

# ==========================================
# 👑 ページ基本設定（ブラウザのタブに表示される文字）
# ==========================================
st.set_page_config(
    page_title="味覚分析AI | Taste Analyzer AI",
    page_icon="🍰",
    layout="centered"
)

# ==========================================
# 🧠 AIモデル読み込み（本家TensorFlow CPU版を使用）
# ==========================================
@st.cache_resource
def load_model():
    try:
        # 👑 相性エラーを回避するため、本家Googleの正式版TensorFlowを使用します
        import tensorflow as tf
        
        if os.path.exists(MODEL_PATH):
            # 本家ツールのInterpreterを使用（互換性エラーを完全回避）
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            return interpreter, input_details, output_details
        else:
            st.error(f"⚠️ AIモデルファイルが見つかりません：{MODEL_PATH}")
    except Exception as e:
        st.error(f"モデル読込エラー: {e}")
    return None, None, None

model_interpreter, input_details, output_details = load_model()

# ==========================================
# 🔮 AI予測処理（画像のサイズを自動調整して判定）
# ==========================================
def predict_taste(pil_image):
    if model_interpreter is None:
        return 0, 0, 0
        
    try:
        # モデルが求める画像のサイズ（縦・横）を自動取得
        input_shape = input_details[0]['shape']
        h, w = input_shape[1], input_shape[2]
        
        # 画像をモデルのサイズに変換して数値化
        img_resized = pil_image.resize((w, h))
        img_array = np.array(img_resized, dtype=np.float32)
        
        # モデルの要求に合わせてデータを 0.0〜1.0 に正規化
        if input_details[0]['dtype'] == np.float32:
            img_array = img_array / 255.0
            
        # バッチ次元の追加
        img_array = np.expand_dims(img_array, axis=0)
        
        # AIに入力して計算開始
        model_interpreter.set_tensor(input_details[0]['index'], img_array)
        model_interpreter.invoke()
        
        # 結果を取得
        output_data = model_interpreter.get_tensor(output_details[0]['index'])[0]
        
        # 出力データから甘味・辛味・酸味を取得
        sweet = float(output_data[0]) if len(output_data) > 0 else 0.0
        spicy = float(output_data[1]) if len(output_data) > 1 else 0.0
        sour = float(output_data[2]) if len(output_data) > 2 else 0.0
        
        # 0〜1の確率なら100倍してパーセントにする
        if max(sweet, spicy, sour) <= 1.0:
            sweet *= 100
            spicy *= 100
            sour *= 100
            
        return min(100, max(0, sweet)), min(100, max(0, spicy)), min(100, max(0, sour))
    except Exception as e:
        st.error(f"予測計算エラー: {e}")
        return 0, 0, 0

# ==========================================
# 🖥️ 画面レイアウト（Streamlit UI）
# ==========================================
st.title("🍰 味覚分析AI アプリ")
st.write("画像をアップロードすると、AIがその見た目から「甘味」「辛味」「酸味」の度合いを分析します！")

# 🌐 多言語切り替え（簡易版）
lang = st.sidebar.selectbox("Language / 言語", ["日本語", "English"])

# 📥 画像アップローダー
uploaded_file = st.file_uploader(
    "画像を選択またはドラッグ＆ドロップしてください" if lang == "日本語" else "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # アップロードされた画像を開く
    image = Image.open(uploaded_file).convert("RGB")
    
    # 2列に分けてレイアウト（左に画像、右に結果）
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="分析する画像" if lang == "日本語" else "Target Image", use_container_width=True)
        
    with col2:
        with st.spinner("AI分析中..." if lang == "日本語" else "Analyzing..."):
            # AIで予測
            sweet, spicy, sour = predict_taste(image)
            
        st.subheader("📊 分析結果" if lang == "日本語" else "📊 Results")
        
        # メーター（プログレスバー）で表示
        st.write(f"🍬 甘味 (Sweet): {sweet:.1f}%")
        st.progress(int(sweet) / 100)
        
        st.write(f"🌶️ 辛味 (Spicy): {spicy:.1f}%")
        st.progress(int(spicy) / 100)
        
        st.write(f"🍋 酸味 (Sour): {sour:.1f}%")
        st.progress(int(sour) / 100)
        
        # 一番数値が高い味のイラストを表示するオマケ機能
        highest_taste = max(sweet, spicy, sour)
        illust_file = None
        
        if highest_taste == sweet and sweet > 10:
            st.success("✨ とても甘そうな食べ物です！" if lang == "日本語" else "✨ Looks very sweet!")
            illust_file = "sweet.png"
        elif highest_taste == spicy and spicy > 10:
            st.error("🔥 辛そうな食べ物です！" if lang == "日本語" else "🔥 Looks very spicy!")
            illust_file = "spicy.png"
        elif highest_taste == sour and sour > 10:
            st.warning("🍋 酸っぱそうな食べ物です！" if lang == "日本語" else "🍋 Looks very sour!")
            illust_file = "sour.png"
            
        # illustフォルダに画像があれば表示
        if illust_file:
            target_illust_path = os.path.join(ILLUST_DIR, illust_file)
            if os.path.exists(target_illust_path):
                st.image(target_illust_path, width=150)
