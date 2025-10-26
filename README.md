# Previsione Sismica ai Campi Flegrei con Deep Learning

<p align="center">
  <strong>Un modello ibrido (CNN-LSTM) per la stima sperimentale del rischio sismico a breve termine, basato su dati multi-fisici.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow Version">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn" alt="Scikit-Learn Version">
  <img src="https://img.shields.io/badge/Status-Sperimentale-red?style=for-the-badge" alt="Project Status">
</p>

---

## 📋 Indice dei Contenuti

1.  [**Contesto e Motivazione**](#-contesto-e-motivazione)
2.  [**Obiettivi del Progetto**](#-obiettivi-del-progetto)
3.  [**La Pipeline in Dettaglio**](#-la-pipeline-in-dettaglio)
4.  [**Architettura del Modello**](#-architettura-del-modello)
5.  [**Setup ed Esecuzione**](#-setup-ed-esecuzione)
6.  [**Disclaimer Fondamentale**](#️-disclaimer-fondamentale)
7.  [**Sviluppi Futuri**](#-sviluppi-futuri)

---

## 🌍 Contesto e Motivazione

I **Campi Flegrei** sono una caldera vulcanica attiva e densamente popolata, soggetta al fenomeno del **bradisismo**: un lento sollevamento del suolo accompagnato da un'intensa attività sismica. La recente escalation di questi fenomeni ha riacceso l'attenzione sulla necessità di strumenti di analisi avanzati.

Questo progetto nasce come un'esplorazione accademica per indagare se le moderne tecniche di Deep Learning possano identificare pattern complessi nei dati geofisici, fornendo un supporto statistico e sperimentale ai sistemi di monitoraggio tradizionali.

---

## 🎯 Obiettivi del Progetto

| Obiettivo Principale                     | Dettagli                                                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Sviluppare un Modello di Classificazione** | Prevedere la probabilità di eventi sismici (`M ≥ 3.5`) in un orizzonte temporale di **7 giorni**.        |
| **Adottare un Approccio Multi-Fisico**     | Integrare dati **sismologici** (INGV) con dati di **deformazione del suolo** (GNSS) per una visione completa. |
| **Creare una Pipeline End-to-End**         | Automatizzare l'intero processo, dall'acquisizione dei dati in tempo reale alla stima finale.           |

---

## 🔧 La Pipeline in Dettaglio

La pipeline è il cuore del progetto e si articola in 4 fasi automatizzate:

| Fase                     | Descrizione                                                                                                                                                                                                                                     |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Acquisizione Dati** | Download del catalogo sismico ufficiale dell'**INGV** e delle serie storiche di deformazione del suolo da stazioni **GNSS** selezionate.                                                                                                            |
| **2. Feature Engineering** | Calcolo di una serie storica giornaliera con feature aggregate su 30 giorni, tra cui **energia sismica**, **b-value** (indicatore di stress) e **velocità di sollevamento** del suolo.                                                               |
| **3. Preparazione Dati** | Creazione della variabile target (evento sì/no), standardizzazione delle feature e trasformazione del dataset in **sequenze temporali** adatte al modello LSTM.                                                                                    |
| **4. Addestramento**     | Training del modello, valutazione su un test set e generazione della stima probabilistica finale utilizzando i dati più recenti disponibili.                                                                                                     |

---

## 🧠 Architettura del Modello

Il modello sfrutta un'architettura ibrida per massimizzare la capacità di apprendimento dalla sequenza di dati temporali.

**Flusso del Modello:**
`Input` ➔ `Conv1D` ➔ `MaxPooling1D` ➔ `LSTM` ➔ `Dense` ➔ `Output (Probabilità)`

-   **Strato `Conv1D`**: Funziona come un estrattore di feature, identificando pattern e correlazioni locali tra le variabili in piccole finestre temporali (es. 3 giorni).
-   **Strato `LSTM`**: Modella le dipendenze temporali a lungo termine, imparando i trend che possono precedere un evento sismico significativo.

Questa struttura ibrida permette di catturare sia le micro-correlazioni sia le macro-tendenze nei dati.

---

## 🚀 Setup ed Esecuzione

### Prerequisiti
-   Python 3.8+
-   Git

### Installazione

1.  **Clona il repository:**
    ```bash
    # Clona il progetto sul tuo computer locale
    git clone https://github.com/giuseppegautieri/Predictor_Campi_flegrei.git
    ```
    ```bash
    # Entra nella cartella
    cd Predictor_Campi_flegrei
    ```

2.  **Crea e attiva un ambiente virtuale:**
    ```bash
    # Crea un ambiente isolato per le dipendenze
    python -m venv venv
    ```
    ```bash
    # Attiva l'ambiente dopo averlo creato
    source venv/bin/activate  # Su Windows: venv\Scripts\activate
    ```

3.  **Installa le dipendenze:**
    ```bash
    # Installa tutte le librerie necessarie
    pip install -r requirements.txt
    ```

### Esecuzione dello Script

Per lanciare l'intera pipeline, esegui semplicemente il file principale:
```bash
# Esegui lo script
python predictor_near_live.py
```

L'output mostrerà i log di ogni fase, i risultati della valutazione e la stima finale.

---

## ⚠️ Disclaimer Fondamentale
<p align="center">
  <strong>ATTENZIONE: QUESTO È UN PROGETTO SPERIMENTALE.</strong>
</p>

 La previsione dei terremoti è un problema scientifico irrisolto. I risultati di questo modello sono basati su correlazioni statistiche e **non hanno alcun valore previsionale deterministico** o **ufficiale**.
 
 Non basare alcuna decisione relativa alla sicurezza personale o pubblica sui risultati di questo script.

---

## 💡 Sviluppi Futuri

- [ ] **Integrazione di Dati Geochimici**: Aggiungere dati sulla composizione dei gas delle fumarole per arricchire il modello.
- [ ] **Ottimizzazione Iperparametri**: Utilizzare Keras Tuner o Optuna per trovare la migliore configurazione del modello in modo automatico.
- [ ] **Deployment di un'API**: Creare un endpoint (es. con FastAPI) per interrogare il modello addestrato e ottenere previsioni on-demand.
- [ ] **Dashboard Interattiva**: Sviluppare un'interfaccia utente con Streamlit o Plotly/Dash per visualizzare i dati storici e i risultati del modello in modo interattivo.




