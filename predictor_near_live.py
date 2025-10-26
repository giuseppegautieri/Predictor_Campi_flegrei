# ===================================================================
# PASSO 0: IMPORTAZIONI E IMPOSTAZIONI GLOBALI
# ===================================================================
# Importazioni di base per la manipolazione dati e la visualizzazione
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from functools import reduce
from datetime import datetime, timedelta

# Importazioni per la preparazione dei dati e la valutazione da Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Importazioni per il modello di Deep Learning con TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input, BatchNormalization, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# Importazioni per l'acquisizione dati sismologici
try:
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    print("ObsPy importato con successo.")
except ImportError:
    print("ATTENZIONE: obspy non trovato. Tentativo di installazione...")
    # Questo blocco è per ambienti come Kaggle/Colab dove le installazioni non sono persistenti
    import sys
    #!{sys.executable} -m pip install "obspy==1.4.2" lxml --quiet
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    print("ObsPy installato e importato.")


# --- Impostazioni dell'Esperimento ---
TIME_WINDOW_DAYS = 30
MAGNITUDE_THRESHOLD = 3.5
PREDICTION_HORIZON_DAYS = 7
SEQUENCE_LENGTH = 30

# --- Parametri per l'acquisizione dati ---
DATA_CENTER_CLIENT = "INGV"
START_DATE_ANALYSIS = "2024-07-01"
# Coordinate Geografiche Precise per i Campi Flegrei
MIN_LAT, MAX_LAT = 40.78, 40.87
MIN_LON, MAX_LON = 14.05, 14.27


# ===================================================================
# PASSO 1: ACQUISIZIONE DATI SISMICI "ON-DEMAND"
# ===================================================================
print("--- [PASSO 1] Acquisizione Dati Sismici 'On-Demand' ---")
try:
    client = Client(DATA_CENTER_CLIENT)
    start_time_utc = UTCDateTime(START_DATE_ANALYSIS)
    end_time_utc = UTCDateTime(datetime.utcnow())
    print(f"Richiesta dati sismici dal {start_time_utc.date.isoformat()} a oggi...")

    # La chiamata standard che usa il formato QuakeML
    catalog = client.get_events(
        starttime=start_time_utc, endtime=end_time_utc,
        minlatitude=MIN_LAT, maxlatitude=MAX_LAT,
        minlongitude=MIN_LON, maxlongitude=MAX_LON
    )
    
    print(f"Trovati {len(catalog)} eventi nel catalogo.")

    # Processa l'oggetto Catalog
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

# Pulizia finale del DataFrame sismico
df_sismic = df_sismic.dropna().sort_values(by='time').reset_index(drop=True)
print(f"Dati sismici pronti: {len(df_sismic)} eventi validi.")
print("--------------------------------------------------\n")


# ===================================================================
# PASSO 2: FEATURE ENGINEERING SISMICO
# ===================================================================
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


# ===================================================================
# PASSO 2.5: INTEGRAZIONE DATI GNSS (DEFORMAZIONE SUOLO)
# ===================================================================
print("--- [PASSO 2.5] Integrazione Dati GNSS (Deformazione Suolo) ---")
# --- MODIFICA CHIAVE: LA TUA LISTA DI STAZIONI VERIFICATE ---
# Inserisci qui i codici delle 5 stazioni che hai trovato.
# Questo è solo un esempio, sostituiscilo con i tuoi codici reali.
stazioni_da_usare = ["TAI1", "MAFE", "NAPO", "FRUL", "POIS"] # ESEMPIO: Sostituisci con le tue stazioni
# --- FINE MODIFICA ---

try:
    print(f"Verrà tentato il download dei dati per le seguenti stazioni: {stazioni_da_usare}")

    lista_df_gnss = []
    # Ciclo per scaricare i dati di ogni stazione nella tua lista
    for stazione in stazioni_da_usare:
        # Costruisce l'URL specifico per ogni stazione
        url_gnss = f"https://geodesy.unr.edu/gps_timeseries/tenv3_loadpredictions/{stazione}.tenv3"
        
        try:
            print(f"Download e parsing dei dati per la stazione '{stazione}'...")
            
            response_stazione = requests.get(url_gnss)
            response_stazione.raise_for_status() # Controlla se la richiesta HTTP ha avuto successo
            
            data_rows = []
            lines = response_stazione.text.splitlines() # Divide il contenuto del file in righe
            
            # Ciclo per processare ogni riga del file di testo
            for line in lines[1:]: # !!! INIZIAMO DALLA SECONDA RIGA PER SALTARE L'INTESTAZIONE MANUALE !!!
                if not line.strip(): # Salta righe vuote
                    continue

                parts = line.split() # Dividi la riga in base agli spazi
                
                # Controlla che la riga abbia abbastanza colonne (almeno 11 per avere data e up)
                if len(parts) > 8: 
                    # Estrae la data (colonna 2, indice 1) e la componente verticale (colonna 11, indice 10)
                    date_str = parts[1]
                    up_component = float(parts[8]) # La componente 'up' è alla 9a colonna (indice 8)
                    data_rows.append({'date_str': date_str, f'sollevamento_{stazione}': up_component})

            if not data_rows:
                print(f"ATTENZIONE: Nessun dato valido trovato per la stazione '{stazione}'. Verrà saltata.")
                continue
                
            df_stazione = pd.DataFrame(data_rows)
            # Converti la colonna della data nel formato corretto (es. '08JUL04')
            df_stazione['date'] = pd.to_datetime(df_stazione['date_str'], format='%y-%b-%d', errors='coerce')
            df_stazione = df_stazione.dropna().drop(columns=['date_str']) # Rimuovi righe con data non valida e la colonna stringa temporanea
            lista_df_gnss.append(df_stazione)
            print(f"Dati per la stazione '{stazione}' processati con successo.")
            
        except requests.exceptions.HTTPError as http_err:
             print(f"ATTENZIONE: Impossibile trovare i dati per la stazione '{stazione}' (Errore HTTP: {http_err}). La stazione verrà saltata.")
        except ValueError as ve: # Cattura specificamente errori di formato dati
             print(f"ATTENZIONE: Errore di parsing per la stazione '{stazione}': {ve}. Potrebbe essere un problema con i dati di quella stazione o con le colonne selezionate. Verrà saltata.")
        except Exception as e:
            print(f"ATTENZIONE: Errore generico nel processare la stazione '{stazione}': {e}. La stazione verrà saltata.")

    if not lista_df_gnss:
        raise ValueError("Nessun dato GNSS valido è stato scaricato da nessuna delle stazioni specificate. Impossibile procedere.")

    # --- Unione dei dati di tutte le stazioni scaricate con successo ---
    # Usa reduce per unire tutti i DataFrame in lista_df_gnss in uno unico
    df_gnss_completo = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), lista_df_gnss)
    df_gnss_completo = df_gnss_completo.sort_values(by='date').reset_index(drop=True)
    
    # --- Interpolazione dei dati GNSS ---
    # Applica l'interpolazione temporale a tutte le colonne di sollevamento
    for col in df_gnss_completo.columns:
        if col != 'date': # Assicurati di non interpolare la colonna 'date'
            df_gnss_completo[col] = df_gnss_completo[col].interpolate(method='time')
    
    # Riempie eventuali valori NaN rimasti (es. all'inizio o alla fine della serie)
    df_gnss_completo = df_gnss_completo.fillna(method='ffill').fillna(method='bfill')
    print("Dati da tutte le stazioni GNSS disponibili uniti e puliti.")

    # --- Unione con il DataFrame delle features sismiche ---
    features_df['date'] = pd.to_datetime(features_df['date'])
    features_df_arricchito = pd.merge(features_df, df_gnss_completo, on='date', how='left')

    # --- Creazione di Features di Deformazione ---
    print("Creazione features di deformazione...")
    colonne_sollevamento = [col for col in features_df_arricchito.columns if col.startswith('sollevamento_')]
    
    # Media del sollevamento tra tutte le stazioni GNSS
    features_df_arricchito['sollevamento_medio_caldera'] = features_df_arricchito[colonne_sollevamento].mean(axis=1)
    
    # Velocità media di sollevamento (derivata)
    features_df_arricchito['velocita_media_sollevamento_7d'] = features_df_arricchito['sollevamento_medio_caldera'].diff(periods=7)
    
    # --- Pulizia Finale ---
    # Manteniamo le colonne individuali per il modello (se si vuole sperimentare)
    # Altrimenti si potrebbero droppare per avere meno features. Per ora le teniamo.
    # Riempiamo eventuali NaN rimasti (es. se una stazione non ha dati in un certo periodo)
    features_df = features_df_arricchito.fillna(0) # Riempiamo con 0 i NaN, presumendo assenza di misura = assenza di deformazione
    
    print("Nuove features di deformazione (basate su più stazioni) create e aggiunte.")
    print("Anteprima del dataset arricchito:")
    print(features_df.head())
    print("Ultime righe del dataset arricchito:")
    print(features_df.tail())

except Exception as e:
    print(f"ATTENZIONE: Fallimento critico nel blocco di integrazione dati GNSS: {e}. Il modello userà solo i dati sismici.")
    # In caso di qualsiasi errore nel download/processamento GNSS, continuiamo solo con i dati sismici
    # Assicurandoci che le colonne per le features GNSS esistano comunque per evitare errori successivi
    features_df['sollevamento_medio_caldera'] = 0
    features_df['velocita_media_sollevamento_7d'] = 0
    # Se ci fossero altre features GNSS, aggiungerle qui a 0

print("--------------------------------------------------\n")


# ===================================================================
# PASSO 3: CREAZIONE TARGET E PREPARAZIONE DATI SEQUENZIALI
# ===================================================================
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


# ===================================================================
# PASSO 4: COSTRUZIONE E ADDESTRAMENTO DEL MODELLO
# ===================================================================
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


# ===================================================================
# PASSO 5: VALUTAZIONE E STIMA FUTURA
# ===================================================================
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