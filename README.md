Previsione Sismica ai Campi Flegrei con Modello Ibrido CNN-LSTM
![alt text](https://www.ingv.it/stampa-e-comunicazione/comunicati-stampa/5135-campi-flegrei-nuovo-studio-sui-meccanismi-di-risalita-del-magma/@@images/58b02da2-1262-4b2a-b605-64539acab2d2.jpeg)

Immagine: INGV
Progetto sperimentale per la previsione a breve termine del rischio sismico nella caldera dei Campi Flegrei, utilizzando un modello di Deep Learning multi-fisico che integra dati sismici e di deformazione del suolo.
📜 Indice
Contesto e Motivazione
Obiettivo del Progetto
✨ Caratteristiche Principali
🔧 Pipeline del Progetto
🧠 Architettura del Modello
🛠️ Stack Tecnologico
⚙️ Installazione e Setup
▶️ Come Eseguire lo Script
📊 Interpretazione dei Risultati
⚠️ Disclaimer Importante
🚀 Sviluppi Futuri
🙏 Ringraziamenti
🌍 Contesto e Motivazione
I Campi Flegrei sono una caldera vulcanica attiva situata a ovest di Napoli, una delle aree a più alto rischio vulcanico al mondo a causa della sua intensa urbanizzazione. L'area è soggetta al fenomeno del bradisismo: un lento sollevamento e abbassamento del suolo accompagnato da un'intensa attività sismica a bassa magnitudo.
Dal 2005, è in corso una nuova fase di sollevamento che ha causato un'escalation della sismicità, generando preoccupazione tra la popolazione e le autorità. Questo progetto nasce come un'esplorazione accademica per verificare se le moderne tecniche di intelligenza artificiale possano contribuire a identificare pattern precursori di eventi sismici di magnitudo "rilevante" (M ≥ 3.5), affiancando i sistemi di monitoraggio tradizionali.
🎯 Obiettivo del Progetto
L'obiettivo è sviluppare un modello di classificazione binaria in grado di stimare la probabilità che si verifichi almeno un evento sismico di magnitudo superiore a una soglia definita (MAGNITUDE_THRESHOLD = 3.5) entro un orizzonte temporale futuro di 7 giorni (PREDICTION_HORIZON_DAYS).
Il modello non si basa solo sulla sismicità passata, ma adotta un approccio multi-fisico, integrando anche i dati sulla deformazione del suolo (GNSS), un parametro fisico fondamentale per descrivere la dinamica del sistema vulcanico.
✨ Caratteristiche Principali
Acquisizione Dati On-Demand: Lo script si collega direttamente ai server dell'INGV e del UNR per scaricare i dati più recenti, garantendo un'analisi sempre aggiornata.
Approccio Multi-Fisico: Fusione di dati sismologici (magnitudo, frequenza, energia) e dati geodetici (sollevamento del suolo) per una visione d'insieme più completa.
Feature Engineering Avanzato: Calcolo di parametri sismologici significativi come il b-value e la velocità di deformazione per catturare le dinamiche del sottosuolo.
Modello Ibrido Deep Learning: Utilizzo di una rete CNN-LSTM per estrarre feature spaziali locali dalle serie temporali e modellarne le dipendenze a lungo termine.
Gestione Classi Sbilanciate: Implementazione di class_weight per gestire la rarità degli eventi sismici rilevanti e addestrare un modello più robusto.
Pipeline End-to-End: Il progetto copre l'intero ciclo di vita: acquisizione dati, pre-processing, training, valutazione e stima predittiva finale.
🔧 Pipeline del Progetto
Il processo è suddiviso in 5 passi logici e automatizzati:
Acquisizione Dati Sismici: Download del catalogo eventi sismici dall'INGV per l'area dei Campi Flegrei.
Feature Engineering Sismico: Creazione di una serie temporale giornaliera con features aggregate (es. n° eventi, max magnitudo, energia, b-value) su una finestra mobile di 30 giorni.
Integrazione Dati GNSS: Download e processamento dei dati di sollevamento del suolo da stazioni GNSS multiple. Creazione di feature aggregate come il sollevamento medio della caldera e la sua velocità.
Preparazione Dati per il Modello: Creazione della variabile target, scaling delle features e trasformazione del dataset in sequenze temporali adatte per il modello LSTM.
Addestramento e Valutazione: Costruzione, compilazione e addestramento del modello CNN-LSTM. Valutazione delle performance su un test set tramite matrice di confusione e classification report.
Stima Futura: Utilizzo dell'ultima sequenza di dati disponibili per generare una stima probabilistica per i successivi 7 giorni.
🧠 Architettura del Modello
Il modello ibrido è progettato per sfruttare i punti di forza di due architetture complementari:
Conv1D (Strato Convoluzionale): Agisce come un estrattore di feature. Analizza la sequenza di input per identificare pattern locali e interazioni tra le diverse features (es. una rapida accelerazione del sollevamento che coincide con un calo del b-value).
LSTM (Long Short-Term Memory): Riceve le feature estratte dalla CNN e modella le dipendenze temporali a lungo termine. Questo permette al modello di "ricordare" trend e pattern che si sviluppano su più giorni o settimane.
Dense (Strati Finali): Interpretano l'output della LSTM per produrre la classificazione finale (probabilità di evento).
🛠️ Stack Tecnologico
Data Analysis & Processing: Pandas, NumPy, Scikit-learn
Deep Learning: TensorFlow, Keras
Acquisizione Dati Sismici: ObsPy
Richieste HTTP: requests
Visualizzazione: Matplotlib, Seaborn
⚙️ Installazione e Setup
Per eseguire questo progetto localmente, segui questi passi:
Clona il repository:
code
Bash
git clone https://github.com/TUO_USERNAME/NOME_REPOSITORY.git
cd NOME_REPOSITORY
Crea un ambiente virtuale (consigliato):
code
Bash
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
Installa le dipendenze:
Assicurati di avere un file requirements.txt. Se non ce l'hai, crealo con il seguente contenuto:
code
Txt
pandas
numpy
matplotlib
seaborn
requests
scikit-learn
tensorflow
obspy
lxml
Quindi, installa le librerie:
code
Bash
pip install -r requirements.txt
▶️ Come Eseguire lo Script
Una volta completato il setup, puoi lanciare l'intera pipeline con un singolo comando:
code
Bash
python nome_del_tuo_script.py
Lo script stamperà a console i log di ogni passo, l'architettura del modello, l'avanzamento dell'addestramento e, infine, i risultati della valutazione e la stima futura.
📊 Interpretazione dei Risultati
L'output finale include:
Metriche sul Test Set: Accuracy, Precision, Recall e F1-Score per valutare le performance del modello.
Matrice di Confusione: Mostra nel dettaglio i successi e gli errori del modello (Veri Positivi/Negativi, Falsi Positivi/Negativi).
Stima Probabilistica: La probabilità stimata di un evento sismico rilevante nei prossimi 7 giorni.
code
Code
--- Stima Sperimentale per il Futuro ---

Probabilità stimata di 'Nessun Evento': 85.10%
Probabilità stimata di 'Evento Rilevante': 14.90%

AVVISO: Questa è una stima statistica sperimentale.
⚠️ Disclaimer Importante
Questo progetto è un'esplorazione accademica e sperimentale e NON deve essere utilizzato per scopi operativi o di protezione civile. La previsione dei terremoti è un problema scientifico complesso e ancora irrisolto. Le stime prodotte da questo modello sono il risultato di correlazioni statistiche e non hanno valore previsionale deterministico. Non basare alcuna decisione relativa alla sicurezza personale su questi risultati.
🚀 Sviluppi Futuri
Integrazione di Dati Aggiuntivi: Aggiungere dati geochimici (es. composizione dei gas fumarolici) o dati satellitari (InSAR) per una visione ancora più completa.
Ottimizzazione degli Iperparametri: Utilizzare tecniche come Keras Tuner o Optuna per trovare la configurazione ottimale del modello.
Architetture Alternative: Sperimentare con modelli più recenti come i Transformer, che si sono dimostrati molto efficaci nell'analisi di sequenze.
Deployment: Creare una semplice web app (es. con Streamlit o Flask) per visualizzare i dati e le previsioni in modo interattivo.
🙏 Ringraziamenti
Questo progetto è stato possibile grazie alla disponibilità di dati aperti forniti da:
Istituto Nazionale di Geofisica e Vulcanologia (INGV) per i dati sismologici.
University of Nevada, Reno (UNR) per i dati geodetici GNSS.
