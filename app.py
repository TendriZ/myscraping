from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from collections import Counter
import re
import io
import json
import time
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Backend sentiment analysis functions (dari kode Python Anda)
def load_comments_from_file_stream(file_obj):
    """Load komentar dari file stream (untuk menghindari file locking)"""
    try:
        # Reset file pointer ke awal
        file_obj.seek(0)
        
        # Baca file langsung dari memory stream
        df = pd.read_excel(file_obj.stream)
        
        comment_columns = ['comment', 'Comment', 'AuthorComment', 'text', 'Text', 'Komentar']
        comment_col = None
        
        for col in comment_columns:
            if col in df.columns:
                comment_col = col
                break
        
        if comment_col is None:
            return []
        
        comments = df[comment_col].dropna().astype(str).tolist()
        return comments
        
    except Exception as e:
        print(f"Error loading from stream: {e}")
        return []

def load_comments_with_retry(file_path, max_retries=3):
    """Load komentar dengan retry mechanism untuk handle file locking"""
    for attempt in range(max_retries):
        try:
            # Tunggu sebentar sebelum retry
            if attempt > 0:
                time.sleep(1)
            
            df = pd.read_excel(file_path)
            
            comment_columns = ['comment', 'Comment', 'AuthorComment', 'text', 'Text', 'Komentar']
            comment_col = None
            
            for col in comment_columns:
                if col in df.columns:
                    comment_col = col
                    break
            
            if comment_col is None:
                return []
            
            comments = df[comment_col].dropna().astype(str).tolist()
            return comments
            
        except PermissionError as e:
            print(f"Attempt {attempt + 1}: File is locked, retrying...")
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                return []
        except Exception as e:
            print(f"Error loading Excel (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return []
    
    return []

def cleanup_file_with_retry(file_path, max_retries=5):
    """Cleanup file dengan retry untuk handle Windows file locking"""
    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File cleaned up successfully: {file_path}")
                return True
        except PermissionError:
            print(f"Cleanup attempt {attempt + 1}: File still locked, retrying in 1 second...")
            time.sleep(1)
        except Exception as e:
            print(f"Cleanup error: {e}")
            break
    
    print(f"Warning: Could not cleanup file {file_path}")
    return False

def load_comments_from_excel(file_path):
    """Load komentar dari file Excel"""
    try:
        df = pd.read_excel(file_path)
        
        comment_columns = ['comment', 'Comment', 'AuthorComment', 'text', 'Text', 'Komentar']
        comment_col = None
        
        for col in comment_columns:
            if col in df.columns:
                comment_col = col
                break
        
        if comment_col is None:
            return []
        
        comments = df[comment_col].dropna().astype(str).tolist()
        return comments
        
    except Exception as e:
        print(f"Error loading Excel: {e}")
        return []

def preprocess(text):
    """Preprocessing teks"""
    if pd.isna(text):
        return []
    
    text = str(text).lower()
    
    # Hapus URL, mention, hashtag
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = [word for word in text.split() if len(word) > 2]
    return words

def calculate_tfidf(processed_comments, max_features=3000, min_df=2):
    """Hitung TF-IDF matrix dengan optimasi memory dan handling data kecil"""
    word_freq = Counter()
    for doc in processed_comments:
        word_freq.update(set(doc))
    
    # Adjust parameters untuk dataset kecil
    N = len(processed_comments)
    if N < 10:
        min_df = 1  # Accept semua kata untuk dataset sangat kecil
        max_features = min(500, max_features)  # Kurangi max features
    elif N < 50:
        min_df = max(1, min_df - 1)  # Reduce min_df
        max_features = min(1000, max_features)
    
    valid_words = [word for word, freq in word_freq.items() if freq >= min_df]
    
    # Fallback jika tidak ada kata valid
    if len(valid_words) == 0:
        print("Warning: No valid words found, using all words")
        valid_words = list(word_freq.keys())
    
    if len(valid_words) > max_features:
        valid_words = [word for word, freq in word_freq.most_common(max_features)]
    
    word_to_idx = {word: i for i, word in enumerate(valid_words)}
    
    print(f"TF-IDF: {N} documents, {len(valid_words)} features, min_df={min_df}")
    
    tfidf = lil_matrix((N, len(valid_words)), dtype=np.float32)
    
    for i, doc in enumerate(processed_comments):
        if not doc:
            continue
                
        word_count = Counter(doc)
        doc_length = len(doc)
        
        for word, count in word_count.items():
            if word not in word_to_idx:
                continue
                
            tf = count / doc_length
            df = word_freq[word]
            idf = np.log(N / (1 + df))
            tfidf[i, word_to_idx[word]] = tf * idf
    
    return tfidf.tocsr(), word_to_idx
    
    for i, doc in enumerate(processed_comments):
        if not doc:
            continue
                
        word_count = Counter(doc)
        doc_length = len(doc)
        
        for word, count in word_count.items():
            if word not in word_to_idx:
                continue
                
            tf = count / doc_length
            df = word_freq[word]
            idf = np.log(N / (1 + df))
            tfidf[i, word_to_idx[word]] = tf * idf
    
    return tfidf.tocsr(), word_to_idx

def kmeans(X, k=3, max_iter=50, random_state=42):
    """K-Means clustering untuk sparse matrix dengan handling data kecil"""
    np.random.seed(random_state)
    
    # Check if we have enough data points
    if X.shape[0] < k:
        print(f"Warning: Data points ({X.shape[0]}) kurang dari jumlah cluster ({k})")
        k = max(1, X.shape[0])  # Adjust k to available data
    
    if hasattr(X, 'toarray'):
        if X.shape[0] > 5000:
            sample_size = min(1000, X.shape[0])
            sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[sample_indices].toarray()
            initial_centroids = X_sample[np.random.choice(X_sample.shape[0], k, replace=False)]
        else:
            X_dense = X.toarray()
            # Fix the sampling issue for small datasets
            if X_dense.shape[0] >= k:
                initial_centroids = X_dense[np.random.choice(X_dense.shape[0], k, replace=False)]
            else:
                # For very small datasets, use all available points
                initial_centroids = X_dense[:k]
                # Pad with duplicates if needed
                while initial_centroids.shape[0] < k:
                    idx_to_duplicate = np.random.choice(X_dense.shape[0])
                    initial_centroids = np.vstack([initial_centroids, X_dense[idx_to_duplicate]])
    else:
        if X.shape[0] >= k:
            initial_centroids = X[np.random.choice(X.shape[0], k, replace=False)]
        else:
            initial_centroids = X[:k]
            while initial_centroids.shape[0] < k:
                idx_to_duplicate = np.random.choice(X.shape[0])
                initial_centroids = np.vstack([initial_centroids, X[idx_to_duplicate]])
    
    centroids = initial_centroids.astype(np.float32)
    
    for iteration in range(max_iter):
        if hasattr(X, 'toarray'):
            distances = []
            batch_size = 1000
            for i in range(0, X.shape[0], batch_size):
                batch = X[i:i+batch_size].toarray().astype(np.float32)
                batch_distances = np.linalg.norm(batch[:, np.newaxis] - centroids, axis=2)
                distances.append(batch_distances)
            distances = np.vstack(distances)
        else:
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        
        labels = np.argmin(distances, axis=1)
        
        new_centroids = []
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                if hasattr(X, 'toarray'):
                    cluster_points = X[mask].toarray().astype(np.float32)
                else:
                    cluster_points = X[mask].astype(np.float32)
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(centroids[i])
        
        new_centroids = np.array(new_centroids)
        
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
            
        centroids = new_centroids
    
    return labels, centroids

def label_sentiments(centroids, word_to_idx):
    """Label cluster berdasarkan lexicon sentiment yang diperluas"""
    # Expanded negative words lexicon
    neg_words = {
        # Kata kasar dan makian
        'anjing', 'babi', 'bangsat', 'bajingan', 'kampret', 'tolol', 'bodoh', 'goblok',
        'sialan', 'setan', 'iblis', 'laknat', 'terkutuk', 'brengsek', 'jahanam',
        
        # Kata destruktif
        'hancur', 'bakar', 'bubar', 'rusak', 'roboh', 'tutup', 'matikan', 'bunuh',
        'penjahat', 'kriminal', 'teroris', 'pembunuh',
        
        # Kata korupsi dan politik negatif
        'korup', 'koruptor', 'pencuri', 'penipu', 'bohong', 'dusta', 'munafik',
        'licik', 'curang', 'khianat', 'pengkhianat',
        
        # Emosi negatif
        'marah', 'benci', 'murka', 'geram', 'kesal', 'jengkel', 'dongkol',
        'kecewa', 'sedih', 'menyesal', 'frustrasi', 'stress', 'depresi',
        
        # Kata buruk umum
        'buruk', 'jelek', 'busuk', 'kotor', 'jorok', 'najis', 'hina',
        'rendah', 'sampah', 'gagal', 'kalah', 'lemah', 'payah'
    }
    
    # Expanded positive words lexicon
    pos_words = {
        # Dukungan dan semangat
        'semangat', 'dukung', 'support', 'backing', 'sokong', 'bantu', 'bantuan',
        'solid', 'kuat', 'hebat', 'luar', 'biasa', 'mantap', 'top', 'bagus',
        
        # Persatuan dan perjuangan
        'bersatu', 'persatu', 'satukan', 'perjuangan', 'perjuangkan', 'lawan',
        'merdeka', 'bebas', 'kebebasan', 'kemerdekaan', 'independen',
        
        # Kata positif umum
        'baik', 'bagus', 'hebat', 'keren', 'amazing', 'fantastic', 'excellent',
        'sukses', 'berhasil', 'menang', 'juara', 'terbaik', 'optimal', 'maksimal',
        
        # Emosi positif
        'senang', 'gembira', 'bahagia', 'suka', 'cinta', 'sayang', 'bangga',
        'optimis', 'harapan', 'percaya', 'yakin', 'confident',
        
        # Spiritual/religius positif
        'lindungi', 'blessing', 'berkah', 'rahmat', 'tuhan', 'allah', 'syukur',
        'alhamdulillah', 'subhanallah', 'mashaallah'
    }
    
    cluster_labels = []
    
    for i, centroid in enumerate(centroids):
        neg_score = sum(centroid[word_to_idx[w]] for w in neg_words if w in word_to_idx)
        pos_score = sum(centroid[word_to_idx[w]] for w in pos_words if w in word_to_idx)
        
        print(f"Cluster {i}: Neg={neg_score:.4f}, Pos={pos_score:.4f}")
        
        # Adjusted threshold untuk lebih sensitif
        if neg_score > pos_score and neg_score > 0.001:  # Lower threshold
            cluster_labels.append('Negatif')
        elif pos_score > neg_score and pos_score > 0.001:
            cluster_labels.append('Positif')
        else:
            cluster_labels.append('Netral')
    
    return cluster_labels

def analyze_sentiment_backend(comments, k_clusters=3):
    """Main sentiment analysis function dengan handling data kecil"""
    # Preprocessing
    processed_comments = [preprocess(c) for c in comments]
    
    # Filter empty comments
    valid_indices = [i for i, doc in enumerate(processed_comments) if len(doc) > 0]
    comments = [comments[i] for i in valid_indices]
    processed_comments = [processed_comments[i] for i in valid_indices]
    
    if len(comments) == 0:
        return None, "Tidak ada komentar valid setelah preprocessing"
    
    # Adjust cluster number untuk dataset kecil
    if len(comments) < k_clusters:
        k_clusters = max(1, len(comments))
        print(f"Adjusting clusters to {k_clusters} due to small dataset")
    
    # Check minimum data requirement
    if len(comments) < 3:
        return None, f"Data terlalu sedikit untuk analisis clustering. Minimum 3 komentar, ditemukan {len(comments)}"
    
    try:
        # TF-IDF Calculation
        tfidf_matrix, word_to_idx = calculate_tfidf(processed_comments)
        
        if tfidf_matrix is None or len(word_to_idx) == 0:
            return None, "Tidak dapat membuat TF-IDF matrix. Data mungkin terlalu sedikit atau tidak valid."
        
        # K-Means Clustering
        labels, centroids = kmeans(tfidf_matrix, k=k_clusters)
        
        # Sentiment Labeling
        cluster_sentiments = label_sentiments(centroids, word_to_idx)
        
        # Map labels to sentiments
        sentiments = [cluster_sentiments[label] for label in labels]
        
        # Create result DataFrame
        df_result = pd.DataFrame({
            'Comment': comments,
            'Sentiment': sentiments,
            'Cluster': labels
        })
        
        return df_result, None
        
    except Exception as e:
        return None, f"Error dalam analisis: {str(e)}"

# Routes
@app.route('/')
def index():
    # Serve the HTML file dari dokumen Anda
    with open('sentiment_analysis.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

@app.route('/upload', methods=['POST'])
def upload_file():
    filepath = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Generate unique filename to avoid conflicts
            import uuid
            unique_filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save file
            file.save(filepath)
            
            # Load comments from Excel using file stream instead of file path
            comments = load_comments_from_file_stream(file)
            
            if not comments:
                # Try reading from saved file with retry mechanism
                comments = load_comments_with_retry(filepath)
            
            if not comments:
                return jsonify({'error': 'Tidak dapat membaca komentar dari file. Pastikan file memiliki kolom Comment/AuthorComment.'}), 400
            
            # Perform sentiment analysis
            df_result, error = analyze_sentiment_backend(comments)
            
            if error:
                return jsonify({'error': error}), 400
            
            # Prepare response data
            sentiment_counts = df_result['Sentiment'].value_counts()
            cluster_counts = df_result['Cluster'].value_counts()
            
            # Sample data untuk tabel (max 50 items)
            sample_data = df_result.head(100).to_dict('records')
            sample_data2 = df_result.tail(100).to_dict('records')
            response_data = {
                'totalComments': len(df_result),
                'sentiments': sentiment_counts.to_dict(),
                'clusters': {f'Cluster {k}': v for k, v in cluster_counts.to_dict().items()},
                'sampleData': [
                    {
                        'comment': row['Comment'][:100] + '...' if len(row['Comment']) > 100 else row['Comment'],
                        'sentiment': row['Sentiment'],
                        'cluster': row['Cluster']
                    }
                    for row in sample_data
                ]
            }
            
            return jsonify(response_data)
        
        else:
            return jsonify({'error': 'File type not allowed. Gunakan .xlsx atau .xls'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    finally:
        # Cleanup uploaded file dengan retry
        if filepath and os.path.exists(filepath):
            cleanup_file_with_retry(filepath)

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Install required packages
    print("Make sure you have installed the required packages:")
    print("pip install flask flask-cors pandas numpy scipy openpyxl")
    
    app.run(debug=True, host='0.0.0.0', port=5000)