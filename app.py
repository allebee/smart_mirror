import streamlit as st
import cv2
import tempfile
import os
import base64
import time
import requests
import json
import numpy as np
from io import BytesIO
from PIL import Image
import pygame
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import threading

# Настройка страницы
st.set_page_config(
    page_title="Умное Зеркало - Анализ кожи",
    page_icon="🪞",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS для улучшения интерфейса и анимации звуковой волны
st.markdown("""
<style>
    .main {
        background-color: #1E1E1E;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .results-area {
        background-color: #2E2E2E;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .title {
        text-align: center;
        font-size: 36px;
        margin-bottom: 30px;
        font-weight: bold;
    }
    .loading {
        text-align: center;
        font-size: 24px;
        margin: 30px 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .thinking-dots {
        display: inline-block;
        width: 30px;
        text-align: left;
        height: 24px;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 12px;
        color: #888;
    }
    .highlight {
        background-color: #4CAF50;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
    }
    /* Audio player styling */
    audio {
        width: 100%;
        margin-top: 15px;
        margin-bottom: 15px;
        border-radius: 30px;
        height: 40px;
        display: none; /* Скрываем стандартный аудиоплеер */
    }
    /* Hide video player controls */
    .stCamera > div {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        overflow: hidden;
    }
    /* Аудио волна */
    .audio-wave-container {
        height: 60px;
        background-color: #2a0854;
        border-radius: 8px;
        padding: 5px;
        margin: 15px 0;
        overflow: hidden;
        position: relative;
    }
    .audio-wave {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
    }
    .wave-bar {
        background: #00f6fe;
        width: 4px;
        margin: 0 2px;
        border-radius: 2px;
        animation: sound 0ms -800ms linear infinite alternate;
    }
    .wave-bar:nth-child(4n+1) {
        background: #ff48b0;
    }
    
    @keyframes sound {
        0% {
            height: 10%;
        }
        100% {
            height: 80%;
        }
    }
    .rec-indicator {
        position: absolute;
        top: 10px;
        right: 10px;
        width: 12px;
        height: 12px;
        background-color: #f44336;
        border-radius: 50%;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    /* Категории рекомендаций */
    .rec-category {
        background-color: #3E3E3E;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid #4CAF50;
        transition: transform 0.2s;
    }
    .rec-category:hover {
        transform: translateX(5px);
    }
    .rec-category h4 {
        margin: 0;
        color: #4CAF50;
    }
    .rec-category p {
        margin: 5px 0 0 0;
    }
</style>
""", unsafe_allow_html=True)

# Инициализация сессии
if 'result' not in st.session_state:
    st.session_state.result = None
if 'image' not in st.session_state:
    st.session_state.image = None
if 'audio_played' not in st.session_state:
    st.session_state.audio_played = False
if 'loading' not in st.session_state:
    st.session_state.loading = False
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'playing_audio' not in st.session_state:
    st.session_state.playing_audio = False
if 'play_time' not in st.session_state:
    st.session_state.play_time = 0

# Функция для кодирования изображения в base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Функция для преобразования текста в речь с помощью OpenAI API
def text_to_speech(text, voice="nova"):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("API ключ OpenAI не найден. Пожалуйста, установите его в настройках.")
        return None
    
    # Проверка на пустой текст
    if not text or text.strip() == "":
        st.error("Пустой текст для преобразования в речь")
        return None
    
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "tts-1",
        "input": text,
        "voice": voice,
        "speed": 1.0
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Ошибка API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Ошибка при запросе к API: {e}")
        return None

# Функция для создания визуализации звуковой волны
def create_audio_wave():
    # Создаем 30 столбцов разной высоты для имитации звуковой волны
    wave_html = '<div class="audio-wave-container"><div class="audio-wave">'
    for i in range(30):
        # Вариируем скорость анимации для более реалистичного эффекта
        animation_duration = np.random.randint(500, 1500)
        animation_delay = np.random.randint(0, 1000)
        height = np.random.randint(30, 90)
        
        wave_html += f'''
        <div class="wave-bar" style="animation-duration: {animation_duration}ms; 
                                     animation-delay: {animation_delay}ms;
                                     height: {height}%;"></div>
        '''
    wave_html += '<div class="rec-indicator"></div></div></div>'
    return wave_html

# Функция для анализа изображения с помощью GPT-4o
def analyze_image(image_path):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "API ключ OpenAI не найден. Пожалуйста, установите его в настройках."

    base64_image = encode_image(image_path)
    
    # Инициализация модели GPT-4o
    model = ChatOpenAI(model="gpt-4o")
    
    # Создание сообщения с текстом и изображением
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": """Представь, что ты эксперт по уходу за кожей, который разговаривает с клиентом. 
                Дай короткий и понятный ответ (не больше 5-6 предложений) о том, что ты видишь на фото лица - тип кожи, текстуру и основные характеристики.
                
                Затем перечисли 3-4 ключевые рекомендации в формате JSON для таблицы с полями:
                1. "категория" (например, "Очищение", "Увлажнение", "Защита", "Питание")
                2. "рекомендация" (короткая конкретная рекомендация)
                
                Ответ должен быть на русском языке, разговорным, дружелюбным тоном, как будто говорит реальный человек. Избегай формальных или длинных предложений.
                
                Формат ответа точно такой:
                [ГОЛОС]
                Твоя короткая речь (5-6 предложений максимум) разговорным тоном
                [/ГОЛОС]
                
                [ТАБЛИЦА]
                [{"категория": "Категория1", "рекомендация": "Рекомендация1"}, {"категория": "Категория2", "рекомендация": "Рекомендация2"}, ...]
                [/ТАБЛИЦА]
                """
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    )
    
    # Получение ответа от модели
    response = model.invoke([message])
    return response.content

# Функция для воспроизведения звука в фоновом режиме
def play_audio_thread(audio_data):
    pygame.mixer.init()
    sound = pygame.mixer.Sound(BytesIO(audio_data))
    st.session_state.play_time = time.time()
    st.session_state.playing_audio = True
    sound.play()
    time.sleep(sound.get_length())
    st.session_state.playing_audio = False

# Заголовок приложения
st.markdown("<div class='title'>🪞 Умное Зеркало: Советы по уходу за кожей</div>", unsafe_allow_html=True)

# Боковая панель для настроек
with st.sidebar:
    st.header("Настройки")
    api_key = st.text_input("OpenAI API ключ", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    voice_option = st.selectbox(
        "Голос для озвучивания",
        ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        index=4
    )
    
    st.subheader("О приложении")
    st.write("""
    Это приложение использует искусственный интеллект для предоставления общих советов по уходу за кожей.
    Оно НЕ предоставляет медицинских диагнозов или лечения и предназначено только для информационных целей.
    Для диагностики проблем с кожей всегда обращайтесь к квалифицированному дерматологу.
    """)

# Проверяем, нужно ли запустить анализ
if st.session_state.loading and not st.session_state.processing_done and st.session_state.image:
    with st.spinner('Анализирую...'):
        st.session_state.result = analyze_image(st.session_state.image)
        st.session_state.loading = False
        st.session_state.audio_played = False
        st.session_state.processing_done = True
        st.rerun()

# Две колонки: для камеры и для результатов
col1, col2 = st.columns([1, 1])

with col1:
    # Камера или загрузка изображения
    option = st.radio("Выберите источник изображения:", ["Камера", "Загрузить фото"])
    
    if option == "Камера":
        # Показываем видеопоток с камеры
        img_file_buffer = st.camera_input("Камера", label_visibility="collapsed", key="camera")
        if img_file_buffer is not None:
            # Сохраняем изображение во временный файл
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg")
            tfile.write(img_file_buffer.getvalue())
            st.session_state.image = tfile.name
            
            if st.button("Анализировать фото", key="analyze_cam"):
                if not os.environ.get("OPENAI_API_KEY"):
                    st.error("Пожалуйста, введите API ключ OpenAI в настройках.")
                else:
                    with st.spinner():
                        st.session_state.loading = True
                        st.session_state.processing_done = False
                        st.rerun()  # Перезагрузка страницы для отображения состояния загрузки
    else:
        # Загрузка изображения
        uploaded_file = st.file_uploader("Выберите изображение для анализа", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Отображаем загруженное изображение
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженное изображение", use_column_width=True)
            
            # Сохраняем во временный файл
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg")
            tfile.write(uploaded_file.getbuffer())
            st.session_state.image = tfile.name
            
            if st.button("Анализировать фото", key="analyze_upload"):
                if not os.environ.get("OPENAI_API_KEY"):
                    st.error("Пожалуйста, введите API ключ OpenAI в настройках.")
                else:
                    with st.spinner():
                        st.session_state.loading = True
                        st.session_state.processing_done = False
                        st.experimental_rerun()  # Перезагрузка страницы для отображения состояния загрузки

with col2:
    # Область для отображения результатов
    if st.session_state.loading:
        st.markdown("<div class='loading'>Анализирую... <div class='thinking-dots'></div></div>", unsafe_allow_html=True)
        # Добавим анимацию "думающих" точек
        st.markdown("""
        <style>
            .thinking-dots:after {
              content: '.';
              animation: dots 1.5s steps(5, end) infinite;
            }
            
            @keyframes dots {
              0%, 20% { content: '.'; }
              40% { content: '..'; }
              60% { content: '...'; }
              80%, 100% { content: ''; }
            }
        </style>
        """, unsafe_allow_html=True)
        
    if st.session_state.result:
        # Разделяем речь и таблицу
        result_text = st.session_state.result
        voice_text = ""
        table_json = ""
        
        # Извлекаем текст для голоса
        if "[ГОЛОС]" in result_text and "[/ГОЛОС]" in result_text:
            voice_start = result_text.find("[ГОЛОС]") + len("[ГОЛОС]")
            voice_end = result_text.find("[/ГОЛОС]")
            voice_text = result_text[voice_start:voice_end].strip()
        
        # Извлекаем JSON для таблицы
        if "[ТАБЛИЦА]" in result_text and "[/ТАБЛИЦА]" in result_text:
            table_start = result_text.find("[ТАБЛИЦА]") + len("[ТАБЛИЦА]")
            table_end = result_text.find("[/ТАБЛИЦА]")
            table_json = result_text[table_start:table_end].strip()
        
        # Отображаем результат речи
        st.markdown("<div class='results-area'>", unsafe_allow_html=True)
        st.markdown("### Результаты анализа:")
        st.markdown(voice_text)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Воспроизведение результата голосом и отображение звуковой волны
        if not st.session_state.audio_played:
            # Генерируем аудио, если его еще нет и текст не пустой
            if st.session_state.audio_data is None and voice_text and voice_text.strip() != "":
                st.session_state.audio_data = text_to_speech(voice_text, voice_option)
                
            if st.session_state.audio_data:
                # Отображаем анимацию звуковой волны
                st.markdown(create_audio_wave(), unsafe_allow_html=True)
                
                # Скрытый аудиоплеер для возможности скачивания
                audio_file = BytesIO(st.session_state.audio_data)
                st.audio(audio_file, format="audio/mp3", autoplay=True)
                
                # Запускаем аудио в фоновом режиме, если оно еще не воспроизводится
                if not st.session_state.playing_audio:
                    audio_thread = threading.Thread(
                        target=play_audio_thread, 
                        args=(st.session_state.audio_data,)
                    )
                    audio_thread.daemon = True
                    audio_thread.start()
                    st.session_state.audio_played = True
        
        # Если воспроизведение закончилось, скрываем волну
        if not st.session_state.playing_audio and st.session_state.audio_played:
            st.markdown("<style>.audio-wave-container { display: none; }</style>", unsafe_allow_html=True)
        
        # Отображаем таблицу с рекомендациями
        try:
            if table_json and table_json.strip():
                try:
                    recommendations = json.loads(table_json)
                    
                    st.markdown("<div class='results-area' style='margin-top: 20px;'>", unsafe_allow_html=True)
                    st.markdown("### Ключевые рекомендации:")
                    
                    # Создаем красивую таблицу с рекомендациями
                    for i, rec in enumerate(recommendations):
                        st.markdown(f"""
                        <div class="rec-category">
                            <h4>{rec['категория']}</h4>
                            <p>{rec['рекомендация']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                except json.JSONDecodeError as e:
                    st.error(f"Ошибка формата JSON: {str(e)}")
                    st.info("Исходный текст рекомендаций:")
                    st.code(table_json)
                    
                    # Попытаемся отобразить рекомендации в простом формате
                    st.markdown("<div class='results-area' style='margin-top: 20px;'>", unsafe_allow_html=True)
                    st.markdown("### Общие рекомендации:")
                    st.write(table_json)
                    st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Не удалось отобразить рекомендации: {e}")

# Сброс результатов
if st.button("Начать заново"):
    if st.session_state.image and os.path.exists(st.session_state.image):
        os.unlink(st.session_state.image)
    st.session_state.result = None
    st.session_state.image = None
    st.session_state.audio_played = False
    st.session_state.audio_data = None
    st.session_state.loading = False
    st.session_state.processing_done = False
    st.session_state.playing_audio = False
    st.experimental_rerun()

# Подвал
st.markdown("<div class='footer'>© 2025 Умное Зеркало. Приложение предоставляет только общие рекомендации по уходу за кожей, а не медицинские советы. Для диагностики и лечения кожных заболеваний необходима консультация дерматолога.</div>", unsafe_allow_html=True)