from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from deepface import DeepFace
import tempfile
import os
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

EMOTION_KEYS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/<path:path>")
def website_files(path):
    # Serve the MoodCam website and its CSS/JS/images from the same HTTPS server.
    # This is important for mobile camera access because getUserMedia requires a secure context.
    if path in {"health", "analyze"}:
        return jsonify({"success": False, "message": "Not found"}), 404
    return send_from_directory(BASE_DIR, path)


@app.get("/health")
def health():
    return jsonify({
        "success": True,
        "message": "MoodCam AI server is running",
        "ai": "DeepFace"
    })


@app.post("/analyze")
def analyze():
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "ไม่พบรูปภาพที่ส่งมา"
        }), 400

    image = request.files["image"]

    if not image or image.filename == "":
        return jsonify({
            "success": False,
            "message": "ไม่ได้เลือกรูปภาพ"
        }), 400

    temp_path = None

    try:
        # Save uploaded image temporarily.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image.save(temp.name)
            temp_path = temp.name

        print("🤖 MoodCam: เริ่มวิเคราะห์ด้วย DeepFace...")

        # Real AI emotion analysis.
        # OpenCV is used as the face detector because it is simple to install
        # and works well for normal front-facing photos.
        result = DeepFace.analyze(
            img_path=temp_path,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=True,
            align=True,
            silent=True
        )

        if isinstance(result, list):
            if not result:
                raise ValueError("ไม่พบผลลัพธ์จาก AI")
            result = result[0]

        emotions = result.get("emotion", {})

        if not emotions:
            raise ValueError("AI ไม่สามารถอ่านค่าอารมณ์จากใบหน้าได้")

        # DeepFace returns emotion scores as percentages (0-100).
        scores = {}
        for key in EMOTION_KEYS:
            scores[key] = round(float(emotions.get(key, 0.0)) / 100.0, 6)

        dominant = max(scores, key=scores.get)

        response = {
            "success": True,
            "result": scores,
            "dominant_emotion": dominant,
            "face_confidence": float(result.get("face_confidence", 0.0))
        }

        print("✅ MoodCam: วิเคราะห์เสร็จแล้ว")
        print(scores)

        return jsonify(response)

    except Exception as e:
        print("❌ MoodCam AI ERROR:")
        traceback.print_exc()

        error_text = str(e)

        # Make common DeepFace errors easier to understand in the browser.
        if "Face could not be detected" in error_text:
            message = "AI ตรวจไม่พบใบหน้า กรุณาใช้รูปที่เห็นใบหน้าชัดเจน"
        elif "No face" in error_text or "face" in error_text.lower():
            message = "AI ไม่สามารถตรวจจับใบหน้าได้ กรุณาใช้ภาพที่เห็นใบหน้าชัดเจน"
        else:
            message = "AI วิเคราะห์ไม่สำเร็จ กรุณาดูข้อความในหน้าต่าง Terminal ของ Python"

        return jsonify({
            "success": False,
            "message": message,
            "error": error_text
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    print("=" * 60)
    print("MoodCam AI Server")
    print("DeepFace Emotion Analysis")
    print("HTTPS: https://127.0.0.1:5000")
    print("LAN:   https://<IP-ของคอม>:5000")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        ssl_context="adhoc"
    )
