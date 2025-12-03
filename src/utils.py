"""
Utilidades generales para el proyecto de clasificación de neumonía.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import json

def setup_project_paths():
    """
    Configura las rutas del proyecto.
    
    Returns:
        dict: Diccionario con las rutas principales del proyecto
    """
    project_root = Path.cwd().parent if 'notebooks' in str(Path.cwd()) else Path.cwd()
    
    paths = {
        'project_root': project_root,
        'data_raw': project_root / 'data' / 'raw',
        'data_processed': project_root / 'data' / 'processed',
        'src': project_root / 'src',
        'notebooks': project_root / 'notebooks',
        'results': project_root / 'results',
        'results_figures': project_root / 'results' / 'figures',
        'results_models': project_root / 'results' / 'models'
    }
    
    # Crear directorios si no existen
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths

def load_config():
    """
    Carga la configuración del proyecto.
    
    Returns:
        dict: Configuración del proyecto
    """
    project_root = Path.cwd().parent if 'notebooks' in str(Path.cwd()) else Path.cwd()
    config_path = project_root / 'config.pkl'
    
    if config_path.exists():
        import pickle
        with open(config_path, 'rb') as f:
            return pickle.load(f)
    else:
        # Configuración por defecto
        return {
            'IMG_SIZE': (224, 224),
            'RANDOM_SEED': 42,
            'PROJECT_ROOT': str(project_root)
        }

def save_config(config):
    """
    Guarda la configuración del proyecto.
    
    Args:
        config (dict): Configuración a guardar
    """
    project_root = Path.cwd().parent if 'notebooks' in str(Path.cwd()) else Path.cwd()
    config_path = project_root / 'config.pkl'
    
    import pickle
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", 
                         save_path=None, figsize=(8, 6)):
    """
    Crea y muestra una matriz de confusión.
    
    Args:
        y_true (array): Etiquetas verdaderas
        y_pred (array): Predicciones
        class_names (list): Nombres de las clases
        title (str): Título del gráfico
        save_path (str): Ruta para guardar la figura (opcional)
        figsize (tuple): Tamaño de la figura
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_roc_curve(y_true, y_proba, title="ROC Curve", save_path=None, figsize=(8, 6)):
    """
    Crea y muestra una curva ROC.
    
    Args:
        y_true (array): Etiquetas verdaderas
        y_proba (array): Probabilidades predichas para la clase positiva
        title (str): Título del gráfico
        save_path (str): Ruta para guardar la figura (opcional)
        figsize (tuple): Tamaño de la figura
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return roc_auc

def create_classification_report_df(y_true, y_pred, class_names):
    """
    Crea un DataFrame con el reporte de clasificación.
    
    Args:
        y_true (array): Etiquetas verdaderas
        y_pred (array): Predicciones
        class_names (list): Nombres de las clases
    
    Returns:
        pd.DataFrame: Reporte de clasificación como DataFrame
    """
    import pandas as pd
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    return df_report

def visualize_image_grid(images, titles=None, cmap='gray', figsize=(15, 10), 
                        cols=4, save_path=None):
    """
    Visualiza una grilla de imágenes.
    
    Args:
        images (list): Lista de imágenes a mostrar
        titles (list): Lista de títulos (opcional)
        cmap (str): Colormap para mostrar las imágenes
        figsize (tuple): Tamaño de la figura
        cols (int): Número de columnas
        save_path (str): Ruta para guardar la figura (opcional)
    """
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Asegurar que axes sea siempre 2D
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_images):
        row, col = divmod(i, cols)
        
        axes[row, col].imshow(images[i], cmap=cmap)
        if titles and i < len(titles):
            axes[row, col].set_title(titles[i])
        axes[row, col].axis('off')
    
    # Ocultar axes vacías
    for i in range(n_images, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def load_processed_data(data_path, split='train'):
    """
    Carga datos preprocesados desde archivos numpy.
    
    Args:
        data_path (Path): Ruta al directorio de datos procesados
        split (str): División del dataset ('train', 'test', 'val')
    
    Returns:
        tuple: (images, labels) o (None, None) si no se encuentran
    """
    images_path = data_path / f'{split}_images.npy'
    labels_path = data_path / f'{split}_labels.npy'
    
    if images_path.exists() and labels_path.exists():
        images = np.load(images_path)
        labels = np.load(labels_path)
        return images, labels
    else:
        print(f"No se encontraron datos procesados para {split}")
        return None, None

def save_results(results, filename, results_path):
    """
    Guarda resultados en formato JSON.
    
    Args:
        results (dict): Diccionario con resultados
        filename (str): Nombre del archivo
        results_path (Path): Ruta donde guardar
    """
    filepath = results_path / filename
    
    # Convertir arrays numpy a listas para JSON
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_converted = convert_numpy(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_converted, f, indent=2)
    
    print(f"Resultados guardados en: {filepath}")

def print_section_header(title, char="=", length=60):
    """
    Imprime un encabezado de sección formateado.
    
    Args:
        title (str): Título de la sección
        char (str): Carácter para el borde
        length (int): Longitud total del encabezado
    """
    border = char * length
    padding = (length - len(title) - 2) // 2
    header = f"{char * padding} {title} {char * padding}"
    
    print(f"\n{border}")
    print(header)
    print(border)