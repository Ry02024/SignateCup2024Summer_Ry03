import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# seabornのデフォルトスタイルを設定
sns.set(style="whitegrid")

def plot_histogram(df, feature, bins=10, title=None, xlabel=None, ylabel='Frequency', figsize=(10, 6)):
    """
    指定された特徴量のヒストグラムをプロットする関数
    
    Parameters:
    df (pd.DataFrame): データフレーム
    feature (str): プロットする特徴量名
    bins (int): ヒストグラムのビンの数（デフォルトは10）
    title (str): グラフのタイトル
    xlabel (str): x軸のラベル
    ylabel (str): y軸のラベル（デフォルトは 'Frequency'）
    figsize (tuple): グラフのサイズ（デフォルトは (10, 6)）
    
    Returns:
    None
    """
    plt.figure(figsize=figsize)
    plt.hist(df[feature].dropna(), bins=bins, color='skyblue', edgecolor='black')
    plt.title(title or f'Distribution of {feature}')
    plt.xlabel(xlabel or feature)
    plt.ylabel(ylabel)
    plt.show()

def plot_bar_chart(df, feature, title=None, xlabel=None, ylabel='Count', figsize=(10, 6)):
    """
    指定されたカテゴリカル特徴量の棒グラフをプロットする関数
    
    Parameters:
    df (pd.DataFrame): データフレーム
    feature (str): プロットする特徴量名
    title (str): グラフのタイトル
    xlabel (str): x軸のラベル
    ylabel (str): y軸のラベル（デフォルトは 'Count'）
    figsize (tuple): グラフのサイズ（デフォルトは (10, 6)）
    
    Returns:
    None
    """
    plt.figure(figsize=figsize)
    sns.countplot(x=feature, data=df, palette='Set2')
    plt.title(title or f'Distribution of {feature}')
    plt.xlabel(xlabel or feature)
    plt.ylabel(ylabel)
    plt.show()

def plot_correlation_matrix(df, features=None, title='Correlation Matrix', cmap='coolwarm', figsize=(12, 8)):
    """
    相関行列をヒートマップで表示する関数
    
    Parameters:
    df (pd.DataFrame): データフレーム
    features (list): 相関を計算する特徴量のリスト（デフォルトはNoneで全特徴量）
    title (str): グラフのタイトル（デフォルトは 'Correlation Matrix'）
    cmap (str): カラーマップの設定（デフォルトは 'coolwarm'）
    figsize (tuple): グラフのサイズ（デフォルトは (12, 8)）
    
    Returns:
    None
    """
    plt.figure(figsize=figsize)
    corr = df[features].corr() if features else df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, square=True)
    plt.title(title)
    plt.show()

def plot_feature_importances(importances, feature_names, title='Feature Importances', figsize=(10, 8)):
    """
    特徴量の重要度を棒グラフで表示する関数
    
    Parameters:
    importances (array-like): 特徴量の重要度
    feature_names (list): 特徴量名のリスト
    title (str): グラフのタイトル（デフォルトは 'Feature Importances'）
    figsize (tuple): グラフのサイズ（デフォルトは (10, 8)）
    
    Returns:
    None
    """
    indices = importances.argsort()[::-1]
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.show()

def plot_elbow_method(sse, title='Elbow Method for Optimal k', xlabel='Number of clusters', ylabel='SSE', figsize=(10, 6)):
    """
    エルボー法をプロットする関数
    
    Parameters:
    sse (list): 各クラスタ数に対するSSEのリスト
    title (str): グラフのタイトル（デフォルトは 'Elbow Method for Optimal k'）
    xlabel (str): x軸のラベル（デフォルトは 'Number of clusters'）
    ylabel (str): y軸のラベル（デフォルトは 'SSE'）
    figsize (tuple): グラフのサイズ（デフォルトは (10, 6)）
    
    Returns:
    None
    """
    plt.figure(figsize=figsize)
    plt.plot(range(1, len(sse) + 1), sse, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_roc_curve(y_true, y_pred_probs, title='ROC Curve', figsize=(8, 6)):
    """
    ROC曲線をプロットする関数
    
    Parameters:
    y_true (array-like): 実際のターゲット値
    y_pred_probs (array-like): 予測確率
    title (str): グラフのタイトル（デフォルトは 'ROC Curve'）
    figsize (tuple): グラフのサイズ（デフォルトは (8, 6)）
    
    Returns:
    None
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_probs, title='Precision-Recall Curve', figsize=(8, 6)):
    """
    Precision-Recall曲線をプロットする関数
    
    Parameters:
    y_true (array-like): 実際のターゲット値
    y_pred_probs (array-like): 予測確率
    title (str): グラフのタイトル（デフォルトは 'Precision-Recall Curve'）
    figsize (tuple): グラフのサイズ（デフォルトは (8, 6)）
    
    Returns:
    None
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    ap_score = average_precision_score(y_true, y_pred_probs)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, label=f'AP = {ap_score:.2f}')
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', figsize=(8, 6)):
    """
    混同行列をプロットする関数
    
    Parameters:
    y_true (array-like): 実際のターゲット値
    y_pred (array-like): 予測値
    title (str): グラフのタイトル（デフォルトは 'Confusion Matrix'）
    figsize (tuple): グラフのサイズ（デフォルトは (8, 6)）
    
    Returns:
    None
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
