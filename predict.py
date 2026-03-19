import joblib
import re
import gradio as gr
import pandas as pd
from underthesea import word_tokenize

# ===============================
# LOAD MODEL (Pipeline)
# ===============================
model = joblib.load("sentiment_model.pkl")

print("Classes:", model.classes_)


# ===============================
# MAP NHÃN TIẾNG VIỆT
# ===============================
label_map = {
    "negative": "Tiêu cực",
    "neutral": "Trung tính",
    "positive": "Tích cực"
}


# ===============================
# PREPROCESS
# ===============================
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    return " ".join(tokens)


# ===============================
# PHÂN TÍCH
# ===============================
def analyze_text(text):

    try:
        if not text.strip():
            return "⚠️ Không có dữ liệu", pd.DataFrame({"x": [], "y": []})

        lines = text.strip().split("\n")

        results_html = ""
        total_chart = {"negative": 0, "neutral": 0, "positive": 0}

        for i, line in enumerate(lines, 1):

            processed = preprocess(line)

            # 🚀 Pipeline: KHÔNG cần vectorizer
            prediction_en = model.predict([processed])[0]
            probs = model.predict_proba([processed])[0]

            # map đúng class
            prob_dict = dict(zip(model.classes_, probs))

            neg = float(prob_dict.get("negative", 0))
            neu = float(prob_dict.get("neutral", 0))
            pos = float(prob_dict.get("positive", 0))

            # cộng dồn
            total_chart["negative"] += neg
            total_chart["neutral"] += neu
            total_chart["positive"] += pos

            # label tiếng Việt
            prediction = label_map.get(prediction_en, prediction_en)

            # màu đúng theo tiếng Anh
            color = {
                "negative": "red",
                "neutral": "orange",
                "positive": "green"
            }.get(prediction_en, "black")

            # HTML
            results_html += f"""
            <div style="padding:10px; margin:5px; border-radius:10px; border:1px solid #ddd;">
                <b>Câu {i}:</b> {line}<br>
                👉 <b style="color:{color}; font-size:16px;">{prediction}</b><br>
                <small>
                Tiêu cực: {neg:.2f} | Trung tính: {neu:.2f} | Tích cực: {pos:.2f}
                </small>
            </div>
            """

            # print terminal
            print("\n======================")
            print("Câu:", line)
            print("Sentiment:", prediction)
            print("Tiêu cực:", round(neg,2))
            print("Trung tính:", round(neu,2))
            print("Tích cực:", round(pos,2))

        # ===============================
        # BIỂU ĐỒ
        # ===============================
        df = pd.DataFrame({
            "x": ["Tiêu cực", "Trung tính", "Tích cực"],
            "y": [
                total_chart["negative"],
                total_chart["neutral"],
                total_chart["positive"]
            ]
        })

        return results_html, df

    except Exception as e:
        import traceback
        print("🔥 ERROR:", e)
        traceback.print_exc()
        return f"Lỗi: {str(e)}", pd.DataFrame({"x": [], "y": []})


# ===============================
# GIAO DIỆN
# ===============================
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🧠 Phân tích cảm xúc văn bản  
        ### (Naive Bayes + TF-IDF)
        """
    )

    with gr.Row():

        with gr.Column(scale=1):

            input_text = gr.Textbox(
                label="📥 Nhập văn bản (mỗi dòng = 1 câu)",
                lines=8,
                placeholder="Ví dụ:\nHôm nay tôi rất vui\nKết quả khiến tôi thất vọng"
            )

            btn = gr.Button("🚀 Phân tích", variant="primary")

        with gr.Column(scale=1):

            output_chart = gr.BarPlot(
                x="x",
                y="y",
                title="📊 Tổng hợp cảm xúc"
            )

    output_html = gr.HTML(label="📋 Kết quả chi tiết")

    btn.click(
        fn=analyze_text,
        inputs=input_text,
        outputs=[output_html, output_chart]
    )


# ===============================
# CHẠY LOCAL
# ===============================
demo.launch(server_name="127.0.0.1", server_port=7860)