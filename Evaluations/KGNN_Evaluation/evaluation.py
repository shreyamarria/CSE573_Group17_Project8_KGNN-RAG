import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, ndcg_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings("ignore")

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

print("="*80)
print("MOVIELENS KGNN RECOMMENDATION SYSTEM - EVALUATION")
print("="*80)

# ============================================================================
# 1. LOAD DATA & REBUILD GRAPH
# ============================================================================
print("\n[1/8] Loading MovieLens data...")
start_time = time.time()

users = pd.read_csv('users.csv')
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

print(f"  ✓ Loaded {len(users):,} users, {len(movies):,} movies, {len(ratings):,} ratings")
print(f"  ⏱ Time: {time.time() - start_time:.2f}s")

# ============================================================================
# 2. TRAIN/TEST SPLIT
# ============================================================================
print("\n[2/8] Creating train/test split (80/20)...")
start_time = time.time()

# Split ratings per user to ensure each user has train and test data
train_ratings = []
test_ratings = []

for user_id in tqdm(ratings['UserID'].unique(), desc="  Splitting ratings"):
    user_ratings = ratings[ratings['UserID'] == user_id]
    # Ensure at least 1 rating in test if user has multiple ratings
    if len(user_ratings) >= 4:
        train, test = train_test_split(user_ratings, test_size=0.2, random_state=42)
        train_ratings.append(train)
        test_ratings.append(test)
    else:
        # If user has very few ratings, keep most in train
        train_ratings.append(user_ratings)

train_df = pd.concat(train_ratings, ignore_index=True)
test_df = pd.concat(test_ratings, ignore_index=True) if test_ratings else pd.DataFrame()

print(f"  ✓ Train set: {len(train_df):,} ratings")
print(f"  ✓ Test set: {len(test_df):,} ratings")
print(f"  ⏱ Time: {time.time() - start_time:.2f}s")

# ============================================================================
# 3. BUILD KNOWLEDGE GRAPH (USING TRAIN DATA ONLY)
# ============================================================================
print("\n[3/8] Building knowledge graph with training data...")
start_time = time.time()
G = nx.DiGraph()

# Occupation mapping
occupation_map = {
    0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
    4: "college/grad student", 5: "customer service", 6: "doctor/health care",
    7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
    11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
    15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer"
}

# Add users
for _, user in users.iterrows():
    user_id = f"user_{user['UserID']}"
    G.add_node(user_id, ntype="user", user_id=int(user['UserID']))
    
    # Add demographic connections
    gender_id = f"gender_{user['Gender']}"
    G.add_node(gender_id, ntype="gender", name=user['Gender'])
    G.add_edge(user_id, gender_id, etype="is_gender")
    
    age_id = f"age_{user['Age'].replace('-', '_').replace('+', 'plus').replace('<', 'under')}"
    G.add_node(age_id, ntype="age_bucket", name=user['Age'])
    G.add_edge(user_id, age_id, etype="age_bucket")
    
    occ_code = int(user['Occupation'])
    occ_name = occupation_map.get(occ_code, "other")
    occ_id = f"occ_{occ_code}"
    G.add_node(occ_id, ntype="occupation", name=occ_name)
    G.add_edge(user_id, occ_id, etype="has_occupation")

# Add movies
for _, movie in movies.iterrows():
    movie_id = f"movie_{movie['MovieID']}"
    G.add_node(movie_id, ntype="movie", movie_id=int(movie['MovieID']),
               title=movie['Title'], year=int(movie['Year']) if pd.notna(movie['Year']) else None)
    
    # Add genres
    if pd.notna(movie['Genres']):
        for genre in movie['Genres'].split('|'):
            genre = genre.strip()
            genre_id = f"genre_{genre.replace(' ', '_').replace('-', '_')}"
            G.add_node(genre_id, ntype="genre", name=genre)
            G.add_edge(movie_id, genre_id, etype="is_genre")
    
    # Add year decade
    if pd.notna(movie['Year']):
        decade = (int(movie['Year']) // 10) * 10
        year_id = f"year_{decade}s"
        G.add_node(year_id, ntype="year", name=f"{decade}s", decade=decade)
        G.add_edge(movie_id, year_id, etype="year_bucket")

# Add training ratings
for _, rating in tqdm(train_df.iterrows(), total=len(train_df), desc="  Adding ratings"):
    user_id = f"user_{rating['UserID']}"
    movie_id = f"movie_{rating['MovieID']}"
    rating_val = float(rating['Rating'])
    
    if user_id in G and movie_id in G:
        if rating_val >= 4.0:
            G.add_edge(user_id, movie_id, etype="rated_high", rating=rating_val)
        elif rating_val <= 2.0:
            G.add_edge(user_id, movie_id, etype="rated_low", rating=rating_val)
        else:
            G.add_edge(user_id, movie_id, etype="rated_medium", rating=rating_val)

print(f"  ✓ Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
print(f"  ⏱ Time: {time.time() - start_time:.2f}s")

# ============================================================================
# 4. IMPLEMENT GRAPH-BASED RECOMMENDATION ALGORITHM
# ============================================================================
print("\n[4/8] Computing recommendations using graph-based collaborative filtering...")
start_time = time.time()

def get_movie_similarity_score(G, movie1, movie2):
    """Calculate similarity between two movies based on shared attributes"""
    score = 0.0
    
    # Get neighbors (genres, years, users who rated)
    try:
        movie1_neighbors = set(G.neighbors(movie1))
        movie2_neighbors = set(G.neighbors(movie2))
        
        # Shared attributes (genres, years)
        shared_attrs = movie1_neighbors & movie2_neighbors
        for attr in shared_attrs:
            if G.nodes[attr]['ntype'] in ['genre', 'year']:
                score += 1.0
        
        # Normalize by total unique attributes
        total_attrs = len(movie1_neighbors | movie2_neighbors)
        if total_attrs > 0:
            score = score / total_attrs
            
    except:
        pass
    
    return score

def recommend_for_user(G, user_id, k=10):
    """Generate top-k recommendations for a user using graph-based collaborative filtering"""
    user_node = f"user_{user_id}"
    
    if user_node not in G:
        return []
    
    # Get movies the user has already rated
    rated_movies = set()
    for neighbor in G.neighbors(user_node):
        if G.nodes[neighbor]['ntype'] == 'movie':
            rated_movies.add(neighbor)
    
    # Get candidate movies (movies not yet rated)
    all_movies = [n for n in G.nodes() if G.nodes[n]['ntype'] == 'movie']
    candidate_movies = [m for m in all_movies if m not in rated_movies]
    
    # Score each candidate
    movie_scores = {}
    
    for candidate in candidate_movies:
        score = 0.0
        
        # 1. Collaborative filtering: users similar to this user who liked the candidate
        candidate_users = []
        for pred in G.predecessors(candidate):
            if G.nodes[pred]['ntype'] == 'user' and pred != user_node:
                edge_data = G.get_edge_data(pred, candidate)
                if edge_data and edge_data.get('etype') == 'rated_high':
                    candidate_users.append(pred)
        
        # Check similarity between this user and users who liked the candidate
        for other_user in candidate_users:
            # Users are similar if they share demographics or have rated similar movies
            shared = 0
            try:
                user_neighbors = set(G.neighbors(user_node))
                other_neighbors = set(G.neighbors(other_user))
                
                # Count shared demographic attributes
                for node in user_neighbors & other_neighbors:
                    if G.nodes[node]['ntype'] in ['gender', 'age_bucket', 'occupation']:
                        shared += 1
                
                # Count co-rated movies with similar ratings
                for node in user_neighbors & other_neighbors:
                    if G.nodes[node]['ntype'] == 'movie':
                        shared += 0.5
                
                score += shared * 0.1
            except:
                pass
        
        # 2. Content-based: similarity to movies the user liked
        for rated_movie in rated_movies:
            edge_data = G.get_edge_data(user_node, rated_movie)
            if edge_data and edge_data.get('etype') == 'rated_high':
                # High rating - add similarity
                sim = get_movie_similarity_score(G, rated_movie, candidate)
                score += sim * 2.0
            elif edge_data and edge_data.get('etype') == 'rated_low':
                # Low rating - subtract similarity (avoid similar movies)
                sim = get_movie_similarity_score(G, rated_movie, candidate)
                score -= sim * 0.5
        
        movie_scores[candidate] = score
    
    # Sort by score and return top-k
    sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
    return [(m.replace('movie_', ''), score) for m, score in sorted_movies[:k]]

# Generate recommendations for all users in test set
user_recommendations = {}
test_users = test_df['UserID'].unique()

for user_id in tqdm(test_users, desc="  Generating recommendations"):
    user_recommendations[user_id] = recommend_for_user(G, user_id, k=100)

print(f"  ✓ Generated recommendations for {len(user_recommendations):,} users")
print(f"  ⏱ Time: {time.time() - start_time:.2f}s")

# ============================================================================
# 5. COMPUTE EVALUATION METRICS
# ============================================================================
print("\n[5/8] Computing evaluation metrics...")
start_time = time.time()

# Prepare ground truth: what did users actually rate highly in test set?
user_actual_high = defaultdict(set)
user_actual_all = defaultdict(list)

for _, row in test_df.iterrows():
    user_id = row['UserID']
    movie_id = row['MovieID']
    rating = row['Rating']
    
    user_actual_all[user_id].append((movie_id, rating))
    
    # Consider rating >= 4.0 as relevant/positive
    if rating >= 4.0:
        user_actual_high[user_id].add(movie_id)

# --- PRECISION@K ---
def precision_at_k(recommendations, actual_set, k):
    """Calculate precision@k"""
    if not recommendations or not actual_set:
        return 0.0
    
    top_k = [int(movie_id) for movie_id, _ in recommendations[:k]]
    relevant_in_topk = len(set(top_k) & actual_set)
    return relevant_in_topk / k if k > 0 else 0.0

p_at_1_scores = []
p_at_3_scores = []
p_at_5_scores = []

for user_id in user_recommendations:
    if user_id in user_actual_high and len(user_actual_high[user_id]) > 0:
        recs = user_recommendations[user_id]
        actual = user_actual_high[user_id]
        
        p_at_1_scores.append(precision_at_k(recs, actual, 1))
        p_at_3_scores.append(precision_at_k(recs, actual, 3))
        p_at_5_scores.append(precision_at_k(recs, actual, 5))

p_at_1 = np.mean(p_at_1_scores) if p_at_1_scores else 0.0
p_at_3 = np.mean(p_at_3_scores) if p_at_3_scores else 0.0
p_at_5 = np.mean(p_at_5_scores) if p_at_5_scores else 0.0

print(f"  ✓ Precision@1: {p_at_1:.4f} (Target: ≥0.45)")
print(f"  ✓ Precision@3: {p_at_3:.4f} (Target: ≥0.40)")
print(f"  ✓ Precision@5: {p_at_5:.4f} (Target: ≥0.32)")

# --- NDCG@5 ---
def ndcg_at_k(recommendations, actual_ratings_dict, k):
    """Calculate NDCG@k"""
    if not recommendations:
        return 0.0
    
    top_k = [int(movie_id) for movie_id, _ in recommendations[:k]]
    
    # Create relevance scores (actual ratings)
    relevance_scores = []
    for movie_id in top_k:
        if movie_id in actual_ratings_dict:
            relevance_scores.append(actual_ratings_dict[movie_id])
        else:
            relevance_scores.append(0.0)  # Not rated = 0 relevance
    
    if sum(relevance_scores) == 0:
        return 0.0
    
    # Use sklearn's ndcg_score
    true_relevance = np.array([relevance_scores])
    scores = np.array([list(range(len(relevance_scores), 0, -1))])  # Descending scores
    
    try:
        return ndcg_score(true_relevance, scores)
    except:
        return 0.0

ndcg_5_scores = []

for user_id in user_recommendations:
    if user_id in user_actual_all:
        recs = user_recommendations[user_id]
        actual_dict = {int(m): r for m, r in user_actual_all[user_id]}
        
        ndcg_5_scores.append(ndcg_at_k(recs, actual_dict, 5))

ndcg_5 = np.mean(ndcg_5_scores) if ndcg_5_scores else 0.0
print(f"  ✓ NDCG@5: {ndcg_5:.4f} (Target: ≥0.35)")

# --- AUC-ROC & AUC-PR ---
# For each user, create binary classification: will they rate highly (≥4) or not?
y_true_all = []
y_scores_all = []

for user_id in user_recommendations:
    if user_id in user_actual_all:
        recs = user_recommendations[user_id]
        actual_dict = {int(m): r for m, r in user_actual_all[user_id]}
        
        for movie_id_str, score in recs[:50]:  # Top 50 recommendations
            movie_id = int(movie_id_str)
            if movie_id in actual_dict:
                actual_rating = actual_dict[movie_id]
                # Binary: 1 if rated ≥4, 0 otherwise
                y_true_all.append(1 if actual_rating >= 4.0 else 0)
                y_scores_all.append(score)

# Compute AUC-ROC
if len(set(y_true_all)) > 1:  # Need both classes
    auc_roc = roc_auc_score(y_true_all, y_scores_all)
    
    # Compute AUC-PR
    precision_curve, recall_curve, _ = precision_recall_curve(y_true_all, y_scores_all)
    auc_pr = auc(recall_curve, precision_curve)
    
    print(f"  ✓ AUC-ROC: {auc_roc:.4f} (Target: ≥0.94)")
    print(f"  ✓ AUC-PR: {auc_pr:.4f} (Target: ≥0.80)")
else:
    auc_roc = 0.0
    auc_pr = 0.0
    print(f"  ✗ AUC-ROC: N/A (insufficient class diversity)")
    print(f"  ✗ AUC-PR: N/A (insufficient class diversity)")

# --- Binary Classification Metrics (at threshold 0.5) ---
# Convert recommendation scores to binary predictions
threshold = 0.5
if y_scores_all:
    # Normalize scores to [0, 1]
    max_score = max(y_scores_all) if y_scores_all else 1.0
    min_score = min(y_scores_all) if y_scores_all else 0.0
    
    if max_score > min_score:
        y_scores_normalized = [(s - min_score) / (max_score - min_score) for s in y_scores_all]
    else:
        y_scores_normalized = y_scores_all
    
    y_pred = [1 if s >= threshold else 0 for s in y_scores_normalized]
    
    accuracy = accuracy_score(y_true_all, y_pred)
    precision = precision_score(y_true_all, y_pred, zero_division=0)
    recall = recall_score(y_true_all, y_pred, zero_division=0)
    f1 = f1_score(y_true_all, y_pred, zero_division=0)
    
    print(f"  ✓ Accuracy: {accuracy:.4f}")
    print(f"  ✓ Precision: {precision:.4f} (Target: ≥0.12)")
    print(f"  ✓ Recall: {recall:.4f} (Target: ≥0.99)")
    print(f"  ✓ F1-Score: {f1:.4f} (Target: ≥0.22)")
else:
    accuracy = precision = recall = f1 = 0.0
    print(f"  ✗ Classification metrics: N/A")

print(f"  ⏱ Time: {time.time() - start_time:.2f}s")

# ============================================================================
# 6. CREATE EVALUATION VISUALIZATIONS
# ============================================================================
print("\n[6/8] Creating evaluation visualizations...")
start_time = time.time()

# --- Visualization 1: Metrics Comparison Bar Chart ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Precision@K
ax1 = axes[0, 0]
metrics_p = ['P@1', 'P@3', 'P@5']
values_p = [p_at_1, p_at_3, p_at_5]
targets_p = [0.45, 0.40, 0.32]

x_pos = np.arange(len(metrics_p))
bars1 = ax1.bar(x_pos - 0.2, values_p, 0.4, label='Actual', color='steelblue', alpha=0.8)
bars2 = ax1.bar(x_pos + 0.2, targets_p, 0.4, label='Target', color='coral', alpha=0.8)

ax1.set_xlabel('Metric', fontsize=11)
ax1.set_ylabel('Score', fontsize=11)
ax1.set_title('Precision@K Metrics', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(metrics_p)
ax1.legend()
ax1.set_ylim(0, max(max(values_p), max(targets_p)) * 1.2)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 2: AUC Metrics
ax2 = axes[0, 1]
metrics_auc = ['AUC-ROC', 'AUC-PR']
values_auc = [auc_roc, auc_pr]
targets_auc = [0.94, 0.80]

x_pos2 = np.arange(len(metrics_auc))
bars3 = ax2.bar(x_pos2 - 0.2, values_auc, 0.4, label='Actual', color='mediumseagreen', alpha=0.8)
bars4 = ax2.bar(x_pos2 + 0.2, targets_auc, 0.4, label='Target', color='coral', alpha=0.8)

ax2.set_xlabel('Metric', fontsize=11)
ax2.set_ylabel('Score', fontsize=11)
ax2.set_title('AUC Metrics', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos2)
ax2.set_xticklabels(metrics_auc)
ax2.legend()
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3, axis='y')

for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 3: Classification Metrics
ax3 = axes[1, 0]
metrics_class = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values_class = [accuracy, precision, recall, f1]
targets_class = [None, 0.12, 0.99, 0.22]

x_pos3 = np.arange(len(metrics_class))
bars5 = ax3.bar(x_pos3, values_class, 0.6, color='mediumpurple', alpha=0.8, label='Actual')

# Add target lines for metrics with targets
for i, target in enumerate(targets_class):
    if target is not None:
        ax3.axhline(y=target, xmin=(i-0.3)/len(metrics_class), xmax=(i+0.7)/len(metrics_class),
                   color='coral', linestyle='--', linewidth=2, label='Target' if i == 1 else '')

ax3.set_xlabel('Metric', fontsize=11)
ax3.set_ylabel('Score', fontsize=11)
ax3.set_title('Binary Classification Metrics (Threshold=0.5)', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos3)
ax3.set_xticklabels(metrics_class)
ax3.legend()
ax3.set_ylim(0, 1.1)
ax3.grid(True, alpha=0.3, axis='y')

for bar in bars5:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 4: NDCG@5
ax4 = axes[1, 1]
bars6 = ax4.bar([0], [ndcg_5], 0.5, color='darkorange', alpha=0.8, label='Actual')
bars7 = ax4.bar([1], [0.35], 0.5, color='coral', alpha=0.8, label='Target')

ax4.set_xlabel('Metric', fontsize=11)
ax4.set_ylabel('Score', fontsize=11)
ax4.set_title('NDCG@5 Ranking Metric', fontsize=12, fontweight='bold')
ax4.set_xticks([0, 1])
ax4.set_xticklabels(['NDCG@5 (Actual)', 'NDCG@5 (Target)'])
ax4.legend()
ax4.set_ylim(0, max(ndcg_5, 0.35) * 1.3)
ax4.grid(True, alpha=0.3, axis='y')

for bar in list(bars6) + list(bars7):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('evaluation_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: evaluation_metrics_comparison.png")
plt.close()

# --- Visualization 2: Precision@K Distribution ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.hist(p_at_1_scores, bins=30, alpha=0.5, label='P@1', color='steelblue')
ax.hist(p_at_3_scores, bins=30, alpha=0.5, label='P@3', color='mediumseagreen')
ax.hist(p_at_5_scores, bins=30, alpha=0.5, label='P@5', color='coral')

ax.set_xlabel('Precision Score', fontsize=11)
ax.set_ylabel('Number of Users', fontsize=11)
ax.set_title('Distribution of Precision@K Scores Across Users', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('precision_distribution.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: precision_distribution.png")
plt.close()

# --- Visualization 3: User Coverage Analysis ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Count how many users got recommendations at different quality levels
p5_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
p5_counts = np.histogram(p_at_5_scores, bins=p5_bins)[0]
p5_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']

colors_gradient = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4']
bars = ax.bar(p5_labels, p5_counts, color=colors_gradient, alpha=0.8, edgecolor='black')

ax.set_xlabel('Precision@5 Range', fontsize=11)
ax.set_ylabel('Number of Users', fontsize=11)
ax.set_title('User Distribution by Recommendation Quality (P@5)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('user_coverage_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: user_coverage_analysis.png")
plt.close()

print(f"  ⏱ Time: {time.time() - start_time:.2f}s")

# ============================================================================
# 7. GENERATE SUMMARY TABLE
# ============================================================================
print("\n[7/8] Generating summary table...")
start_time = time.time()

summary_data = {
    'Metric': [
        'Precision@1', 'Precision@3', 'Precision@5',
        'AUC-ROC', 'AUC-PR',
        'Accuracy', 'Precision', 'Recall', 'F1-Score',
        'NDCG@5'
    ],
    'Actual Value': [
        f'{p_at_1:.4f}', f'{p_at_3:.4f}', f'{p_at_5:.4f}',
        f'{auc_roc:.4f}', f'{auc_pr:.4f}',
        f'{accuracy:.4f}', f'{precision:.4f}', f'{recall:.4f}', f'{f1:.4f}',
        f'{ndcg_5:.4f}'
    ],
    'Target Value': [
        '≥0.45', '≥0.40', '≥0.32',
        '≥0.94', '≥0.80',
        'N/A', '≥0.12', '≥0.99', '≥0.22',
        '≥0.35'
    ],
    'Status': [
        '✓' if p_at_1 >= 0.45 else '✗',
        '✓' if p_at_3 >= 0.40 else '✗',
        '✓' if p_at_5 >= 0.32 else '✗',
        '✓' if auc_roc >= 0.94 else '✗',
        '✓' if auc_pr >= 0.80 else '✗',
        '—',
        '✓' if precision >= 0.12 else '✗',
        '✓' if recall >= 0.99 else '✗',
        '✓' if f1 >= 0.22 else '✗',
        '✓' if ndcg_5 >= 0.35 else '✗'
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('evaluation_summary.csv', index=False)
print("  ✓ Saved: evaluation_summary.csv")
print(f"  ⏱ Time: {time.time() - start_time:.2f}s")

# ============================================================================
# 8. PRINT FINAL SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("EVALUATION SUMMARY REPORT")
print("="*80)

print("\n📊 RECOMMENDATION QUALITY METRICS:")
print(f"  • Precision@1:  {p_at_1:.4f} {'✓ PASS' if p_at_1 >= 0.45 else '✗ FAIL'} (Target: ≥0.45)")
print(f"  • Precision@3:  {p_at_3:.4f} {'✓ PASS' if p_at_3 >= 0.40 else '✗ FAIL'} (Target: ≥0.40)")
print(f"  • Precision@5:  {p_at_5:.4f} {'✓ PASS' if p_at_5 >= 0.32 else '✗ FAIL'} (Target: ≥0.32)")
print(f"  • NDCG@5:       {ndcg_5:.4f} {'✓ PASS' if ndcg_5 >= 0.35 else '✗ FAIL'} (Target: ≥0.35)")

print("\n📈 RANKING & DISCRIMINATION METRICS:")
print(f"  • AUC-ROC:      {auc_roc:.4f} {'✓ PASS' if auc_roc >= 0.94 else '✗ FAIL'} (Target: ≥0.94)")
print(f"  • AUC-PR:       {auc_pr:.4f} {'✓ PASS' if auc_pr >= 0.80 else '✗ FAIL'} (Target: ≥0.80)")

print("\n🎯 BINARY CLASSIFICATION METRICS (Threshold=0.5):")
print(f"  • Accuracy:     {accuracy:.4f}")
print(f"  • Precision:    {precision:.4f} {'✓ PASS' if precision >= 0.12 else '✗ FAIL'} (Target: ≥0.12)")
print(f"  • Recall:       {recall:.4f} {'✓ PASS' if recall >= 0.99 else '✗ FAIL'} (Target: ≥0.99)")
print(f"  • F1-Score:     {f1:.4f} {'✓ PASS' if f1 >= 0.22 else '✗ FAIL'} (Target: ≥0.22)")

print("\n📁 GENERATED FILES:")
print("  ✓ evaluation_metrics_comparison.png")
print("  ✓ precision_distribution.png")
print("  ✓ user_coverage_analysis.png")
print("  ✓ evaluation_summary.csv")

targets_met = sum([
    p_at_1 >= 0.45, p_at_3 >= 0.40, p_at_5 >= 0.32,
    auc_roc >= 0.94, auc_pr >= 0.80,
    precision >= 0.12, recall >= 0.99, f1 >= 0.22,
    ndcg_5 >= 0.35
])
total_targets = 9

print(f"\n🎯 OVERALL: {targets_met}/{total_targets} target metrics achieved ({targets_met/total_targets*100:.1f}%)")

print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)

