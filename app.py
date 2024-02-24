from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)

# Folder untuk menyimpan file yang di-upload
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder tersebut ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Cek ada file yang di-upload atau tidak
        if 'file' not in request.files:
            return 'No file part', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        
        if file:
            # Simpan file yang di-upload
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Buka file menggunakan Pandas
            data = pd.read_csv(filepath)
            
            # Anggap kita mempunyai model yang sudah di-train sebelumnya
            # dan tersimpan dengan nama 'model.pkl'
            from sklearn.externals import joblib
            model = joblib.load('model.pkl')  # Load model

            # Prediksi skor (contoh menggunakan RandomForestClassifier)
            # Pastikan data cocok dengan model yang digunakan
            predictions = model.predict(data)

            # Simpan skor ke file (misalnya dalam format CSV)
            scores_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'scores.csv')
            pd.DataFrame(predictions).to_csv(scores_filepath, index=False)

            # Tampilkan skor di web
            return jsonify(predictions.tolist())

    # Jika bukan POST, tampilkan form upload
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
