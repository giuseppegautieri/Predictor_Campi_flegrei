import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from functools import reduce
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input, BatchNormalization, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

try:
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    print("ObsPy importato con successo.")
except ImportError:
    print("ATTENZIONE: obspy non trovato. Tentativo di installazione...")
    import sys
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    print("ObsPy installato e importato.")


TIME_WINDOW_DAYS = 30
MAGNITUDE_THRESHOLD = 3.5
PREDICTION_HORIZON_DAYS = 7
SEQUENCE_LENGTH = 30

DATA_CENTER_CLIENT = "INGV"
START_DATE_ANALYSIS = "2024-07-01"
MIN_LAT, MAX_LAT = 40.78, 40.87
MIN_LON, MAX_LON = 14.05, 14.27

print("--- [PASSO 1] Acquisizione Dati Sismici 'On-Demand' ---")
try:
    client = Client(DATA_CENTER_CLIENT)
    start_time_utc = UTCDateTime(START_DATE_ANALYSIS)
    end_time_utc = UTCDateTime(datetime.utcnow())
    print(f"Richiesta dati sismici dal {start_time_utc.date.isoformat()} a oggi...")

    catalog = client.get_events(
        starttime=start_time_utc, endtime=end_time_utc,
        minlatitude=MIN_LAT, maxlatitude=MAX_LAT,
        minlongitude=MIN_LON, maxlongitude=MAX_LON
    )
    
    print(f"Trovati {len(catalog)} eventi nel catalogo.")

    events_data = []
    for event in catalog:
        origin = event.preferred_origin() or event.origins[0]
        magnitude = event.preferred_magnitude() or event.magnitudes[0]
        
        if origin and magnitude:
            events_data.append({
                'time': origin.time.datetime,
                'magnitude': magnitude.mag
            })
            
    if not events_data:
        print("Nessun evento sismico valido trovato. Lo script si interromperà.")
        exit()

    df_sismic = pd.DataFrame(events_data)
    
except Exception as e:
    print(f"ERRORE critico durante il download dei dati sismici: {e}. Impossibile continuare.")
    exit()

df_sismic = df_sismic.dropna().sort_values(by='time').reset_index(drop=True)
print(f"Dati sismici pronti: {len(df_sismic)} eventi validi.")
print("--------------------------------------------------\n")


print("--- [PASSO 2] Feature Engineering Sismico ---")
def calculate_energy(magnitude): return 10**(1.5 * magnitude + 4.8)
features_list = []
date_range = pd.date_range(start=df_sismic['time'].min().date(), end=df_sismic['time'].max().date())
for day in date_range:
    start_window = day - pd.Timedelta(days=TIME_WINDOW_DAYS)
    end_window = day
    subset = df_sismic[(df_sismic['time'] >= start_window) & (df_sismic['time'] < end_window)]
    num_events, max_mag, mean_mag, energy_log, b_value = 0, 0, 0, 0, 0
    if not subset.empty:
        num_events = len(subset); max_mag = subset['magnitude'].max(); mean_mag = subset['magnitude'].mean()
        energy_sum = calculate_energy(subset['magnitude']).sum(); energy_log = np.log10(energy_sum) if energy_sum > 0 else 0
        m_mean, m_min = subset['magnitude'].mean(), subset['magnitude'].min()
        if (m_mean - m_min) > 0: b_value = 0.4343 / (m_mean - m_min)
    features_list.append({'date': day, 'num_events_30d': num_events, 'max_mag_30d': max_mag, 'mean_mag_30d': mean_mag, 'energy_log_30d': energy_log, 'b_value_30d': b_value})
features_df = pd.DataFrame(features_list)
print(f"Creato dataset di features sismiche per {len(features_df)} giorni.")
print("--------------------------------------------------\n")


print("--- [PASSO 2.5] Integrazione Dati GNSS (Deformazione Suolo) ---")
stazioni_da_usare = ["TAI1", "MAFE", "NAPO", "FRUL", "POIS"] 

try:
    print(f"Verrà tentato il download dei dati per le seguenti stazioni: {stazioni_da_usare}")

    lista_df_gnss = []
    for stazione in stazioni_da_usare:
        url_gnss = f"https://geodesy.unr.edu/gps_timeseries/tenv3_loadpredictions/{stazione}.tenv3"
        
        try:
            print(f"Download e parsing dei dati per la stazione '{stazione}'...")
            
            response_stazione = requests.get(url_gnss)
            response_stazione.raise_for_status() 
            data_rows = []
            lines = response_stazione.text.splitlines() 
            
            for line in lines[1:]: 
                if not line.strip(): 
                    continue

                parts = line.split() 
                
                if len(parts) > 8: 
                    date_str = parts[1]
                    up_component = float(parts[8]) 
                    data_rows.append({'date_str': date_str, f'sollevamento_{stazione}': up_component})

            if not data_rows:
                print(f"ATTENZIONE: Nessun dato valido trovato per la stazione '{stazione}'. Verrà saltata.")
                continue
                
            df_stazione = pd.DataFrame(data_rows)
            df_stazione['date'] = pd.to_datetime(df_stazione['date_str'], format='%y-%b-%d', errors='coerce')
            df_stazione = df_stazione.dropna().drop(columns=['date_str']) 
            lista_df_gnss.append(df_stazione)
            print(f"Dati per la stazione '{stazione}' processati con successo.")
            
        except requests.exceptions.HTTPError as http_err:
             print(f"ATTENZIONE: Impossibile trovare i dati per la stazione '{stazione}' (Errore HTTP: {http_err}). La stazione verrà saltata.")
        except ValueError as ve:
             print(f"ATTENZIONE: Errore di parsing per la stazione '{stazione}': {ve}. Potrebbe essere un problema con i dati di quella stazione o con le colonne selezionate. Verrà saltata.")
        except Exception as e:
            print(f"ATTENZIONE: Errore generico nel processare la stazione '{stazione}': {e}. La stazione verrà saltata.")

    if not lista_df_gnss:
        raise ValueError("Nessun dato GNSS valido è stato scaricato da nessuna delle stazioni specificate. Impossibile procedere.")

    df_gnss_completo = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), lista_df_gnss)
    df_gnss_completo = df_gnss_completo.sort_values(by='date').reset_index(drop=True)
    
    for col in df_gnss_completo.columns:
        if col != 'date':
            df_gnss_completo[col] = df_gnss_completo[col].interpolate(method='time')
    
    df_gnss_completo = df_gnss_completo.fillna(method='ffill').fillna(method='bfill')
    print("Dati da tutte le stazioni GNSS disponibili uniti e puliti.")

    features_df['date'] = pd.to_datetime(features_df['date'])
    features_df_arricchito = pd.merge(features_df, df_gnss_completo, on='date', how='left')

    print("Creazione features di deformazione...")
    colonne_sollevamento = [col for col in features_df_arricchito.columns if col.startswith('sollevamento_')]

    features_df_arricchito['sollevamento_medio_caldera'] = features_df_arricchito[colonne_sollevamento].mean(axis=1)

    features_df_arricchito['velocita_media_sollevamento_7d'] = features_df_arricchito['sollevamento_medio_caldera'].diff(periods=7)
    
    features_df = features_df_arricchito.fillna(0) 
    
    print("Nuove features di deformazione (basate su più stazioni) create e aggiunte.")
    print("Anteprima del dataset arricchito:")
    print(features_df.head())
    print("Ultime righe del dataset arricchito:")
    print(features_df.tail())

except Exception as e:
    print(f"ATTENZIONE: Fallimento critico nel blocco di integrazione dati GNSS: {e}. Il modello userà solo i dati sismici.")
    features_df['sollevamento_medio_caldera'] = 0
    features_df['velocita_media_sollevamento_7d'] = 0

print("--------------------------------------------------\n")

print("--- [PASSO 3] Creazione Variabile Target e Preparazione Dati Sequenziali ---")
target_list = []
for index, row in features_df.iterrows():
    current_date = row['date']
    start_prediction = current_date; end_prediction = current_date + pd.Timedelta(days=PREDICTION_HORIZON_DAYS)
    target_event = df_sismic[(df_sismic['time'] >= start_prediction) & (df_sismic['time'] < end_prediction) & (df_sismic['magnitude'] >= MAGNITUDE_THRESHOLD)]
    target_list.append(1 if not target_event.empty else 0)
features_df['target'] = target_list
print(f"Distribuzione del target:\n{features_df['target'].value_counts()}")

feature_columns = [col for col in features_df.columns if col not in ['date', 'target']]
X_features = features_df[feature_columns]
y_labels = features_df['target']
scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_features)
X_sequences, y_sequences = [], []
for i in range(len(X_scaled) - SEQUENCE_LENGTH):
    X_sequences.append(X_scaled[i:i + SEQUENCE_LENGTH])
    y_sequences.append(y_labels.iloc[i + SEQUENCE_LENGTH])
X_sequences = np.array(X_sequences); y_sequences = np.array(y_sequences)
print(f"\nTrasformazione in sequenze completata. Formato dati input: {X_sequences.shape}")
print("--------------------------------------------------\n")

print("--- [PASSO 4] Costruzione e Addestramento del Modello Ibrido CNN-LSTM ---")
X_train, X_test, y_train, y_test = train_test_split(
    X_sequences, y_sequences, test_size=0.20, random_state=42, stratify=y_sequences
)
if len(np.bincount(y_train)) > 1:
    neg, pos = np.bincount(y_train); total = neg + pos
    class_weight = {0: (1 / neg) * (total / 2.0), 1: (1 / pos) * (total / 2.0)}
else:
    class_weight = {0: 1, 1: 1}
print(f"Pesi calcolati per le classi: {class_weight}")

model_cnn_lstm = Sequential([
    Input(shape=(X_sequences.shape[1], X_sequences.shape[2])),
    Conv1D(filters=64, kernel_size=3, activation='silu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2), #Dropout(0.3),
    LSTM(units=50, return_sequences=False), #Dropout(0.3),
    LayerNormalization(),
    Dense(units=25, activation='silu'),
    Dense(units=1, activation='sigmoid')
])
model_cnn_lstm.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)
model_cnn_lstm.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
save_t = TensorBoard()
history = model_cnn_lstm.fit(
    X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
    callbacks=[early_stopping, save_t], class_weight=class_weight, verbose=1
)
print("Addestramento completato.")
print("--------------------------------------------------\n")

print("--- [PASSO 5] Valutazione e Stima Futura con Modello Multi-Fisico ---")
results = model_cnn_lstm.evaluate(X_test, y_test, verbose=0)
print("Risultati Finali sul Test Set:")
for name, value in zip(model_cnn_lstm.metrics_names, results):
    print(f"- {name.capitalize()}: {value:.4f}")
y_pred_proba = model_cnn_lstm.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)
print("\nMatrice di Confusione:"); print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:"); print(classification_report(y_test, y_pred, target_names=['Nessun Evento', 'Evento Rilevante']))
print("\n--- Stima Sperimentale per il Futuro ---")
last_sequence_scaled = X_scaled[-SEQUENCE_LENGTH:]
last_sequence_reshaped = np.expand_dims(last_sequence_scaled, axis=0)
prediction_proba = model_cnn_lstm.predict(last_sequence_reshaped)
print(f"\nProbabilità stimata di 'Nessun Evento': {1 - prediction_proba[0][0]:.2%}")
print(f"Probabilità stimata di 'Evento Rilevante': {prediction_proba[0][0]:.2%}")
print("\nAVVISO: Questa è una stima statistica sperimentale.")
print("--------------------------------------------------\n")
