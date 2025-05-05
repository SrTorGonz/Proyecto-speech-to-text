import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pyunpack import Archive
import matplotlib.pyplot as plt
import tarfile

# Configuración para uso de GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Limitar el uso de memoria de la GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} GPUs físicas, {len(logical_gpus)} GPUs lógicas disponibles")
        # Configurar estrategia de distribución
        strategy = tf.distribute.MirroredStrategy()
        print(f'Número de dispositivos: {strategy.num_replicas_in_sync}')
    except RuntimeError as e:
        print(e)
else:
    print("No se detectaron GPUs, se usará CPU")
    strategy = tf.distribute.get_strategy()

def monitor_gpu_usage():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    print("\n=== Uso de GPU ===")
    for x in local_device_protos:
        if x.device_type == 'GPU':
            print(f"Dispositivo: {x.name}")
            print(f"  Memoria total: {x.memory_limit / (1024**2):.2f} MB")
            print(f"  Tipo: {x.device_type}")
    print("==================")

monitor_gpu_usage()

# Configura tus rutas aquí
dataset_path = 'corpus-dataset.tar.gz'  # Ajusta el nombre si es diferente
# Actualiza la ruta de extracción
extract_dir = os.path.join('common_voice_data', 'cv-corpus-21.0-delta-2025-03-14', 'es')

# Verifica nuevamente
clips_dir = os.path.join(extract_dir, 'clips')
if not os.path.exists(clips_dir):
    print("Estructura incorrecta. Directorios encontrados:")
    print(os.listdir('common_voice_data'))

# Verificar si el archivo existe
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"No se encontró el archivo {dataset_path}. Por favor verifica la ruta.")

# Crear directorio de extracción si no existe
os.makedirs(extract_dir, exist_ok=True)

# Extraer usando tarfile (nativo en Python, no requiere dependencias externas)
try:
    print(f"Extrayendo {dataset_path}...")
    with tarfile.open(dataset_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)
    print("Extracción completada!")
except Exception as e:
    print(f"Error al extraer: {e}")
    raise

# Cargar los metadatos
validated_df = pd.read_csv(os.path.join(extract_dir, 'other.tsv'), sep='\t')

# Filtrar datos válidos y limpiar
validated_df = validated_df[validated_df['path'].notna() & validated_df['sentence'].notna()]
validated_df = validated_df[['path', 'sentence']]

# Preprocesamiento de texto
def preprocess_text(text):
    text = text.lower().strip()
    # Mantener solo caracteres permitidos (ajustar según necesidades)
    allowed_chars = "abcdefghijklmnñopqrstuvwxyz áéíóúü"
    text = ''.join(c for c in text if c in allowed_chars)
    return text

validated_df['sentence'] = validated_df['sentence'].apply(preprocess_text)

# Crear vocabulario
chars = sorted(list(set("".join(validated_df['sentence'].values))))
char_to_num = {c: i for i, c in enumerate(chars)}
num_to_char = {i: c for i, c in enumerate(chars)}

# Parámetros
max_text_len = 150  # Longitud máxima de texto
sample_rate = 16000  # Frecuencia de muestreo
duration = 4  # Duración máxima en segundos
hop_length = 160  # Para cálculo de características
n_mels = 80  # Número de bandas mel

# Función para extraer características MFCC
def extract_features(audio_path):
    try:
        audio, _ = librosa.load(audio_path, sr=sample_rate, duration=duration)
        # Rellenar con ceros si es más corto que la duración esperada
        if len(audio) < sample_rate * duration:
            audio = np.pad(audio, (0, max(0, sample_rate * duration - len(audio))), 
                          mode='constant')
        
        # Extraer características log-mel
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_mels=n_mels, 
            n_fft=400, hop_length=hop_length)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalizar
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        return log_mel.T  # Transponer para tener (time_steps, n_mels)
    except:
        return None

# Función para procesar texto
def text_to_num(text):
    text = preprocess_text(text)
    encoded = [char_to_num[c] for c in text if c in char_to_num]
    return encoded

# Preparar datos
def prepare_data(df, max_samples=10000):
    audio_paths = []
    texts = []
    features = []
    labels = []
    
    for idx, row in df.iterrows():
        if idx >= max_samples:
            break
            
        audio_path = os.path.join(extract_dir, 'clips', row['path'])
        if not os.path.exists(audio_path):
            continue
            
        feature = extract_features(audio_path)
        if feature is None:
            continue
            
        text_encoded = text_to_num(row['sentence'])
        if not text_encoded:
            continue
            
        audio_paths.append(audio_path)
        texts.append(row['sentence'])
        features.append(feature)
        labels.append(text_encoded)
    
    return audio_paths, texts, features, labels

# Limitar a 10000 muestras para entrenamiento rápido
audio_paths, texts, features, labels = prepare_data(validated_df, max_samples=10000)

# Padding para características y etiquetas
X = pad_sequences(features, dtype='float32', padding='post')
y = pad_sequences(labels, maxlen=max_text_len, padding='post', value=len(chars))

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Visualización de ejemplos de datos
print("\n=== Ejemplos de Datos para Entrenamiento ===")
print(f"Número total de muestras: {len(audio_paths)}")
print(f"Forma de los features de audio (X): {X.shape}")
print(f"Forma de las etiquetas de texto (y): {y.shape}")

# Mostrar algunos ejemplos
print("\nEjemplos de audio y texto:")
for i in range(5):
    print(f"\nEjemplo {i+1}:")
    print(f"  Ruta del audio: {audio_paths[i]}")
    print(f"  Texto original: {texts[i]}")
    print(f"  Texto codificado: {labels[i]}")
    print(f"  Forma del feature de audio: {features[i].shape}")
    
    # Decodificar el texto codificado
    decoded_text = ''.join([num_to_char[num] for num in labels[i] if num in num_to_char])
    print(f"  Texto decodificado: {decoded_text}")



# Función para visualizar espectrogramas
def plot_mel_spectrogram(spectrogram, title):
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram.T, aspect='auto', origin='lower', 
              cmap='viridis', vmin=-2, vmax=2)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Tiempo (frames)')
    plt.ylabel('Frecuencia (bandas MEL)')
    plt.tight_layout()
    plt.show()

# Visualizar 3 espectrogramas de ejemplo con sus textos
print("\n=== Visualización de Espectrogramas MEL ===")
for i in range(3):
    spectrogram = features[i]
    text = texts[i]
    print(f"\nEjemplo {i+1}: Texto = '{text}'")
    plot_mel_spectrogram(spectrogram, f"Espectrograma MEL - '{text[:30]}...'")

#CONSTRUCCIÓN DEL MODELO

def build_model(input_shape, num_chars):
    # Entrada
    input_spectrogram = layers.Input(shape=input_shape, name='input_spectrogram')
    
    # Capas convolucionales (inspiradas en ResNet)
    x = layers.Reshape((input_shape[0], input_shape[1], 1))(input_spectrogram)
    
    # Bloque 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Bloque 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Bloque 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((1, 2))(x)
    
    # Redimensionar para RNN
    x = layers.Reshape((-1, 128))(x)
    
    # Capas recurrentes (Bi-LSTM)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    
    # Capa densa
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Capa de salida
    output = layers.Dense(num_chars + 1, activation='softmax')(x)
    
    # Modelo
    model = models.Model(inputs=input_spectrogram, outputs=output, name='speech_to_text')
    
    # Función de pérdida CTC
    def ctc_loss(y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    
    # Compilar
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                 loss=ctc_loss)
    
    return model

# Construir modelo
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_model(input_shape, len(chars))
model.summary()



# ENTRENAMIENTO DEL MODELO

# Callbacks
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    'speech_to_text_model.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False
)

# Entrenamiento
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=30,
    callbacks=[checkpoint_callback]
)



#GUARDAR EL MODELO
import pickle

# Guardar el modelo completo
model.save('speech_to_text_final_model.h5')

# Guardar los mapeos de caracteres y parámetros
with open('speech_to_text_params.pkl', 'wb') as f:
    pickle.dump({
        'char_to_num': char_to_num,
        'num_to_char': num_to_char,
        'max_text_len': max_text_len,
        'sample_rate': sample_rate,
        'duration': duration,
        'hop_length': hop_length,
        'n_mels': n_mels
    }, f)