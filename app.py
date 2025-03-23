import os
import uuid
import cv2
import gradio as gr
import pandas as pd
import numpy as np
from PIL import Image

import roop.globals
from roop.core import start, decode_execution_providers
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import normalize_output_path
from roop.face_analyser import get_one_face

feedback_list = []
session_meta = {}

def get_image_size(path):
    img = Image.open(path)
    return img.width, img.height

def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return None
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def swap_face(source_file, target_file, doFaceEnhancer):
    session_id = str(uuid.uuid4())
    session_dir = os.path.join("temp", session_id)
    os.makedirs(session_dir, exist_ok=True)

    source_path = os.path.join(session_dir, "source.jpg")
    target_path = os.path.join(session_dir, "target.jpg")
    output_path = os.path.join(session_dir, "output.jpg")

    Image.fromarray(source_file).save(source_path)
    Image.fromarray(target_file).save(target_path)

    source_img_cv = cv2.imread(source_path)
    target_img_cv = cv2.imread(target_path)

    source_face = get_one_face(source_img_cv)
    if source_face is None:
        raise gr.exceptions.Error("No face in source image detected.")
    target_face = get_one_face(target_img_cv)
    if target_face is None:
        raise gr.exceptions.Error("No face in target image detected.")

    normalized_output_path = normalize_output_path(source_path, target_path, output_path)
    frame_processors = ["face_enhancer", "face_swapper"] if doFaceEnhancer else ["face_swapper"]

    for frame_processor in get_frame_processors_modules(frame_processors):
        if not frame_processor.pre_check():
            raise gr.exceptions.Error(f"Pre-check failed for {frame_processor}")

    roop.globals.source_path = source_path
    roop.globals.target_path = target_path
    roop.globals.output_path = normalized_output_path
    roop.globals.frame_processors = frame_processors
    roop.globals.headless = True
    roop.globals.keep_fps = True
    roop.globals.keep_audio = True
    roop.globals.keep_frames = False
    roop.globals.many_faces = False
    roop.globals.video_encoder = "libx264"
    roop.globals.video_quality = 18
    roop.globals.execution_providers = decode_execution_providers(['cpu'])
    roop.globals.reference_face_position = 0
    roop.globals.similar_face_distance = 0.6
    roop.globals.max_memory = 60
    roop.globals.execution_threads = 8

    start()

    # Размеры изображений
    sw, sh = get_image_size(source_path)
    tw, th = get_image_size(target_path)
    rw, rh = get_image_size(normalized_output_path)

    # Эмбеддинги и сходство через roop
    result_img_cv = cv2.imread(normalized_output_path)
    result_face = get_one_face(result_img_cv)

    sim_src_tgt = cosine_similarity(source_face.embedding, target_face.embedding)
    sim_src_out = cosine_similarity(source_face.embedding, result_face.embedding if result_face else None)

    # Сохраняем в мета-словарь
    session_meta[session_id] = {
        "source_w": sw, "source_h": sh,
        "target_w": tw, "target_h": th,
        "result_w": rw, "result_h": rh,
        "similar_src_tgt": sim_src_tgt,
        "similar_src_out": sim_src_out
    }

    return normalized_output_path, session_id, gr.update(visible=True), gr.update(visible=True, interactive=False), gr.update(visible=True), gr.update(visible=False)

def submit_feedback(rating_str, session_id):
    rating_num = len(rating_str.strip())
    meta = session_meta.get(session_id, {})

    feedback_list.append({
        "uuid": session_id,
        "rating": rating_num,
        "source_w": meta.get("source_w"),
        "source_h": meta.get("source_h"),
        "target_w": meta.get("target_w"),
        "target_h": meta.get("target_h"),
        "result_w": meta.get("result_w"),
        "result_h": meta.get("result_h"),
        "similar_src_tgt": meta.get("similar_src_tgt"),
        "similar_src_out": meta.get("similar_src_out")
    })

    df = pd.DataFrame(feedback_list)
    df.to_csv("feedback.csv", index=False, encoding="utf-8")

    return (
        "Спасибо за вашу оценку!",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True)
    )


with gr.Blocks() as app:
    gr.Markdown("## 🧠 Замена лиц с оценкой результата и сбором метаданных")

    with gr.Row():
        source_img = gr.Image(label="Источник")
        target_img = gr.Image(label="Цель")

    enhance_checkbox = gr.Checkbox(label="Face Enhancer?")

    # Кнопки: Сгенерировать и Отправить — в одной строке
    with gr.Row():
        generate_btn = gr.Button("🔁 Заменить лицо")
        submit_btn = gr.Button("✅ Отправить отзыв", visible=False, interactive=False)

    rating = gr.Radio(
        choices=["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
        label="Насколько хорошо сработала замена?",
        visible=False
    )

    thanks_msg = gr.Markdown(visible=False)
    output_img = gr.Image(label="Результат")
    session_id_var = gr.Textbox(visible=False)

    instruction_text = gr.Markdown(
        "**Пожалуйста, оцените результат, чтобы продолжить генерацию**",
        visible=False
    )


    # Все клики и связи остались прежними
    generate_btn.click(
        fn=swap_face,
        inputs=[source_img, target_img, enhance_checkbox],
        outputs=[output_img, session_id_var, rating, submit_btn, instruction_text, generate_btn]
    )

    rating.change(
        fn=lambda x: gr.update(interactive=True) if x else gr.update(interactive=False),
        inputs=rating,
        outputs=submit_btn
    )

    submit_btn.click(
        fn=submit_feedback,
        inputs=[rating, session_id_var],
        outputs=[thanks_msg, rating, submit_btn, instruction_text, generate_btn]
    )

from pyngrok import ngrok
public_url = ngrok.connect(7860)
print("🔗 Публичная ссылка:", public_url)
app.launch(server_name="0.0.0.0", server_port=7860)