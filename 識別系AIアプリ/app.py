import os
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import translators as ts

# ==========================================
# 🔧 初期設定とキャッシュ
# ==========================================
st.set_page_config(page_title="AI料理味覚分析アプリ", layout="wide")

# ==========================================
# フォルダ・パス設定
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 👑 os.path.joinを使って、app.pyと同じフォルダ内から探すように修正します
MODEL_PATH = os.path.join(BASE_DIR, "model.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
ILLUST_DIR = os.path.join(BASE_DIR, "illust")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ==========================================
# 🧠 AIモデル読み込み（キャッシュ化して高速化）
# ==========================================
@st.cache_resource
def load_model():
    try:
        import tflite_runtime.interpreter as tflite
        if os.path.exists(MODEL_PATH):
            # 💡 【追加】実際のファイルサイズを計算して画面に出す
            file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            st.info(f"📂 サーバー上の model.tflite のサイズ: {file_size_mb:.3f} MB")
            
            if file_size_mb < 0.01: # 10KB未満など極端に小さい場合
                st.error("⚠️ ファイルサイズが小さすぎます！実データではなく、身代わりファイル（ポインタ）が上がっている可能性があります。")
                return None, None, None

            interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            return interpreter, input_details, output_details
        else:
            st.error(f"⚠️ AIモデルファイルが見つかりません：\n\n{MODEL_PATH}")
    except Exception as e:
        st.error(f"モデル読込エラー: {e}")
    return None, None, None
model_interpreter, input_details, output_details = load_model()

# ==========================================
# 🌐 翻訳・言語マスターデータ
# ==========================================
LANG_DATA = {
    "ja": "日本語", "en": "English", "ko": "한국어", "th": "ภาษาไทย",
    "zh-CN": "简体中文", "es": "Español", "fr": "Français", "it": "Italiano",
    "de": "Deutsch", "vi": "Tiếng Việt", "id": "Bahasa Indonesia"
}

@st.cache_data
def translate_text(text: str, target_code: str) -> str:
    """翻訳処理（同じテキスト・言語の組み合わせはキャッシュして高速化）"""
    if target_code == "ja" or not text:
        return text
    try:
        translated = ts.translate_text(text, from_language='ja', to_language=target_code, translator='google')
        if "hard" in translated.lower():
            translated = translated.lower().replace("hard", "spicy")
        return translated
    except Exception:
        return text

# サバイバルフレーズ辞書（簡略化のため一部のみ表示、元の長文辞書をそのままコピーしてOKです）
SURVIVAL_PHRASES = {
    "辛い": {
        "ja": "「これ、赤くないですが本当に辛くないですか？少しでも辛い成分が入っていれば教えてください」",
        "en": "\"This doesn't look red, but is it really not spicy? Please tell me if it contains even a tiny bit of chili.\"",
        # ... Tkinter版のデータをすべてここにペーストしてください
    },
    "甘い": {
        "ja": "「これは肉/野菜料理ですが、砂糖がたくさん入っていてデザートのように甘い味付けですか？」",
        "en": "\"Is this cooked with a lot of sugar and tastes sweet like a dessert?\"",
        # ... 
    },
    "酸っぱい": {
        "ja": "「これはかなり酸っぱい味付けの料理ですか？」",
        "en": "\"Is this dish very sour?\"",
        # ...
    }
}

# ==========================================
# 🧠 分析ロジック
# ==========================================
def predict_from_pil_image(pil_img: Image.Image) -> tuple:
    if model_interpreter is None:
        return 0, 0, 0
    try:
        size = (224, 224)
        image = ImageOps.fit(pil_img, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.expand_dims(normalized_image_array, axis=0)
        
        model_interpreter.set_tensor(input_details[0]['index'], data)
        model_interpreter.invoke()
        prediction = model_interpreter.get_tensor(output_details[0]['index'])
        pred_scores = prediction[0]
        
        sweet = int(pred_scores[0] * 100) if len(pred_scores) > 0 else 0
        spicy = int(pred_scores[1] * 100) if len(pred_scores) > 1 else 0
        sour  = int(pred_scores[2] * 100) if len(pred_scores) > 2 else 0
        return sweet, spicy, sour
    except Exception as e:
        st.error(f"推論エラー: {e}")
        return 0, 0, 0

def make_analysis_result(sweet: int, spicy: int, sour: int) -> dict:
    max_percent = max(spicy, sweet, sour)
    if max_percent == 0:
        return None

    if spicy == max_percent:
        taste_focus, illust_file = "辛い", "karai.png"
    elif sweet == max_percent:
        taste_focus, illust_file = "甘い", "amai.png"
    else:
        taste_focus, illust_file = "酸っぱい", "suppai.png"

    if max_percent >= 90:
        ai_mode = "大確信"
        summary = f"🔥【超・大確信モード発動】\n\n見なさい！私の電子の眼が凄まじい『{taste_focus}』の波動をキャッチしました！100億%『{taste_focus}』です。"
    elif 65 <= max_percent <= 89:
        ai_mode = "通常"
        summary = f"✨【通常・ちょっとインテリモード】\n\n確率{max_percent}%で『{taste_focus}』の成分が含まれている模様です。お水を用意して実食に臨んでください。"
    elif 40 <= max_percent <= 64:
        ai_mode = "優柔不定"
        summary = f"💦【優柔不安・責任転嫁モード】\n\n一応データ上は{max_percent}%で『{taste_focus}』っぽい気がします。見た目が怪しすぎて自信がありません！"
    else:
        ai_mode = "パニック"
        summary = f"🌀【システム大パニックモード】\n\nすべてのメーターが拮抗していて脳内回路がショートしました！現地人に聞いたほうが早いです！"

    return {
        "spicy": spicy, "sweet": sweet, "sour": sour,
        "mode": ai_mode, "focus": taste_focus, "summary": summary, "illust": illust_file
    }

# ==========================================
# 🎨 Streamlit UI
# ==========================================
# 言語選択の逆引き辞書
NATIVE_TO_CODE = {v: k for k, v in LANG_DATA.items()}

# ヘッダー
col_title, col_lang = st.columns([3, 1])
with col_lang:
    selected_lang_native = st.selectbox("🌐 UI Language", list(LANG_DATA.values()), index=0)
    current_lang = NATIVE_TO_CODE[selected_lang_native]

with col_title:
    st.title(translate_text("海外旅行料理味覚分析AIアシスタント", current_lang))

st.markdown("---")

# メインレイアウト（左：結果、右：画像入力）
col_left, col_right = st.columns([1, 1])

# セッションステートの初期化
if 'analysis_res' not in st.session_state:
    st.session_state.analysis_res = None
if 'uploaded_img' not in st.session_state:
    st.session_state.uploaded_img = None

with col_right:
    st.subheader(translate_text("📸 画像の入力", current_lang))
    
    # タブで「ファイルアップロード」と「カメラ」を分ける
    tab_file, tab_cam = st.tabs([translate_text("📁 画像を選ぶ", current_lang), translate_text("🎥 カメラで撮る", current_lang)])
    
    img_to_process = None
    
    with tab_file:
        uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'webp'])
        if uploaded_file is not None:
            img_to_process = Image.open(uploaded_file).convert("RGB")
            
    with tab_cam:
        camera_img = st.camera_input("")
        if camera_img is not None:
            img_to_process = Image.open(camera_img).convert("RGB")
            
    if img_to_process:
        st.session_state.uploaded_img = img_to_process
        sweet, spicy, sour = predict_from_pil_image(img_to_process)
        st.session_state.analysis_res = make_analysis_result(sweet, spicy, sour)

    # イラスト表示
    if st.session_state.analysis_res:
        res = st.session_state.analysis_res
        illust_path = os.path.join(ILLUST_DIR, res['illust'])
        if os.path.exists(illust_path):
            st.image(illust_path, width=200)

with col_left:
    st.subheader(translate_text("📊 分析結果", current_lang))
    
    if st.session_state.analysis_res is None:
        st.info(translate_text("画像を入力して分析を開始してください。", current_lang))
    else:
        res = st.session_state.analysis_res
        
        # 確率メーター
        st.markdown(f"**🌶 {translate_text('辛い確率', current_lang)}: {res['spicy']}%**")
        st.progress(res['spicy'] / 100)
        
        st.markdown(f"**🍰 {translate_text('甘い確率', current_lang)}: {res['sweet']}%**")
        st.progress(res['sweet'] / 100)
        
        st.markdown(f"**🍋 {translate_text('酸っぱい確率', current_lang)}: {res['sour']}%**")
        st.progress(res['sour'] / 100)
        
        st.markdown("---")
        
        # AIコメント
        st.markdown(f"##### {translate_text('【AIの分析レポート】', current_lang)}")
        st.write(translate_text(res['summary'], current_lang))
        
        # サバイバルフレーズ
        st.markdown("---")
        col_ph_title, col_ph_lang = st.columns([1, 1])
        with col_ph_title:
            st.markdown(f"##### {translate_text('【サバイバルフレーズ】', current_lang)}")
        with col_ph_lang:
            selected_phrase_native = st.selectbox("Phrase Language", list(LANG_DATA.values()), index=0, key="phrase_lang")
            phrase_lang = NATIVE_TO_CODE[selected_phrase_native]
        
        focus = res['focus']
        phrase = SURVIVAL_PHRASES.get(focus, {}).get(phrase_lang, "Please communicate using available languages.")
        st.info(phrase)
