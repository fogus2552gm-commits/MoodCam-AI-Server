from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from deepface import DeepFace
import tempfile
import os
import time
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

DETECTOR_BACKEND = "opencv"

# ---------------------------------------------------------------------------
# Warm up DeepFace models when the process starts (not on the first request).
#
# On Render's free tier, disk is wiped every time the service restarts or
# spins down/up, so DeepFace has to re-download its model + face-detector
# weights. If that happens lazily on the first /analyze call, the request
# can time out or fail mid-download, and the raised exception often contains
# the word "face" (e.g. "haarcascade_frontalface_default.xml"), which used
# to be misreported to users as "no face detected" even though the real
# problem was a failed/slow model download.
#
# Doing it once here, at import time, means the download happens while the
# server is starting up (visible in the Render deploy logs) instead of
# during a real user's request.
# ---------------------------------------------------------------------------
def warm_up_models():
    try:
        print("🔧 MoodCam: กำลังโหลด/ดาวน์โหลดโมเดล DeepFace ล่วงหน้า...")
        start = time.time()
        DeepFace.build_model("Emotion")
        # Also force the face detector backend's weights to download now.
        blank_path = os.path.join(tempfile.gettempdir(), "moodcam_warmup.jpg")
        _make_blank_image(blank_path)
        try:
            DeepFace.analyze(
                img_path=blank_path,
                actions=["emotion"],
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                silent=True
            )
        finally:
            if os.path.exists(blank_path):
                os.remove(blank_path)
        print(f"✅ MoodCam: โหลดโมเดลสำเร็จ ({time.time() - start:.1f}s)")
    except Exception:
        # Don't crash the whole server if warm-up fails; the first real
        # request will simply pay the download cost instead (and we'll
        # still classify the resulting error correctly, see analyze()).
        print("⚠️ MoodCam: โหลดโมเดลล่วงหน้าไม่สำเร็จ (จะลองใหม่ตอนมี request จริง)")
        traceback.print_exc()


def _make_blank_image(path):
    # Tiny neutral-gray image, just enough for DeepFace to initialize its
    # pipeline and download weights. enforce_detection=False means it's
    # fine that no face is present in it.
    from PIL import Image
    Image.new("RGB", (100, 100), color=(128, 128, 128)).save(path)


warm_up_models()


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

        result = _analyze_with_fallback(temp_path)

        if isinstance(result, list):
            if not result:
                raise ValueError("ไม่พบผลลัพธ์จาก AI")
            result = result[0]

        emotions = result.get("emotion", {})

        if not emotions:
            raise ValueError("__NO_EMOTION_DATA__")

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

        message = _classify_error(e)

        return jsonify({
            "success": False,
            "message": message,
            "error": str(e)
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _analyze_with_fallback(temp_path):
    """
    Try strict face detection first. Only if DeepFace genuinely cannot find
    a face do we retry once with enforce_detection=False so a slightly
    imperfect photo (odd angle, busy background, watermark, etc.) still
    gets a best-effort result instead of a hard failure.
    """
    try:
        return DeepFace.analyze(
            img_path=temp_path,
            actions=["emotion"],
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
            silent=True
        )
    except ValueError as e:
        if "Face could not be detected" not in str(e):
            raise
        print("⚠️ MoodCam: ตรวจจับใบหน้าแบบเข้มงวดไม่สำเร็จ กำลังลองแบบผ่อนปรน...")
        return DeepFace.analyze(
            img_path=temp_path,
            actions=["emotion"],
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
            silent=True
        )


def _classify_error(e):
    """
    Turn a raw DeepFace/TensorFlow exception into an accurate Thai message.
    IMPORTANT: this used to match ANY error containing the word "face"
    (e.g. from the file name "haarcascade_frontalface_default.xml"), which
    misreported model-download/timeout failures as "no face detected".
    Now we only use the face-specific message for the exact DeepFace error
    that really means "no face found in the image".
    """
    error_text = str(e)

    if error_text == "__NO_EMOTION_DATA__":
        return "AI ไม่สามารถอ่านค่าอารมณ์จากใบหน้าได้ กรุณาลองรูปอื่น"

    if "Face could not be detected" in error_text:
        return "AI ตรวจไม่พบใบหน้าในภาพ กรุณาใช้รูปที่เห็นใบหน้าชัดเจน แสงเพียงพอ และไม่มีสิ่งบดบัง"

    if "urlopen" in error_text or "URLError" in error_text or "ConnectionError" in error_text:
        return "เซิร์ฟเวอร์ดาวน์โหลดโมเดล AI ไม่สำเร็จ (ปัญหาเครือข่าย) กรุณาลองใหม่อีกครั้งในอีกสักครู่"

    if "Memory" in error_text or "OOM" in error_text or "killed" in error_text.lower():
        return "เซิร์ฟเวอร์หน่วยความจำไม่พอสำหรับวิเคราะห์ กรุณาลองรูปที่มีขนาดเล็กลง หรือลองใหม่อีกครั้ง"

    # Generic fallback for anything unexpected (do NOT assume it's about faces).
    return "AI วิเคราะห์ไม่สำเร็จเนื่องจากปัญหาที่เซิร์ฟเวอร์ กรุณาลองใหม่อีกครั้ง (ดูรายละเอียดใน Logs)"


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
