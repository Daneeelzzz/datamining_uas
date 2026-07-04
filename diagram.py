import graphviz

def buat_diagram_alur():
    dot = graphviz.Digraph(comment='Alur Sistem NLP Random Forest', format='png')
    dot.attr(rankdir='TB', size='8,8')

    # Mendefinisikan Node
    dot.node('A', 'Input: Teks Berita Mentah\n(CSV Dataset)', shape='cylinder')
    dot.node('B', 'Preprocessing:\n- Case Folding\n- Stopword Removal\n- Stemming (Sastrawi)', shape='box')
    dot.node('C', 'Feature Extraction:\nTF-IDF Vectorizer', shape='box')
    dot.node('D', 'Train-Test Split\n(80% Train, 20% Test)', shape='diamond')
    dot.node('E', 'Model Training:\nRandom Forest Classifier', shape='box')
    dot.node('F', 'Model Evaluation:\nAccuracy & Confusion Matrix', shape='box')
    dot.node('G', 'Output: Label Kategori & Probabilitas', shape='ellipse')

    # Mendefinisikan Edge (Alur)
    dot.edges(['AB', 'BC', 'CD'])
    dot.edge('D', 'E', label=' Data Latih')
    dot.edge('E', 'F', label=' Model Terlatih')
    dot.edge('D', 'F', label=' Data Uji')
    dot.edge('F', 'G')

    # Render dan simpan file
    dot.render('alur_sistem_nlp', view=False)
    print("[SUCCESS] Gambar diagram 'alur_sistem_nlp.png' berhasil dibuat!")

if __name__ == '__main__':
    buat_diagram_alur()