"""
Script para descargar el dataset de radiografías de tórax desde Kaggle.
"""

import os
import shutil
from pathlib import Path
import kagglehub

def download_chest_xray_dataset(target_dir=None, force_download=False):
    """
    Descarga el dataset de radiografías de tórax desde Kaggle usando kagglehub.
    
    Args:
        target_dir (str/Path): Directorio objetivo para guardar los datos
        force_download (bool): Si forzar la descarga aunque ya existan los datos
    
    Returns:
        str: Ruta al directorio del dataset descargado
    """
    # Configurar directorio objetivo
    if target_dir is None:
        project_root = Path.cwd()
        target_dir = project_root / 'data' / 'raw'
    else:
        target_dir = Path(target_dir)
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Ruta final donde estará el dataset
    final_dataset_path = target_dir / 'chest_xray'
    
    # Verificar si ya existe el dataset
    if final_dataset_path.exists() and not force_download:
        print(f"✅ Dataset ya existe en: {final_dataset_path}")
        
        # Verificar que tiene la estructura esperada
        required_dirs = ['train/NORMAL', 'train/PNEUMONIA', 
                        'test/NORMAL', 'test/PNEUMONIA',
                        'val/NORMAL', 'val/PNEUMONIA']
        
        all_dirs_exist = all((final_dataset_path / dir_path).exists() 
                           for dir_path in required_dirs)
        
        if all_dirs_exist:
            print("✅ Estructura del dataset verificada.")
            return str(final_dataset_path)
        else:
            print("⚠️ Estructura incompleta, procediendo con la descarga...")
    
    try:
        print("🔄 Descargando dataset desde Kaggle...")
        print("Esto puede tomar varios minutos dependiendo de tu conexión...")
        
        # Descargar usando kagglehub
        downloaded_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print(f"📁 Dataset descargado temporalmente en: {downloaded_path}")
        
        # Buscar la carpeta chest_xray en el directorio descargado
        downloaded_path = Path(downloaded_path)
        chest_xray_source = None
        
        # Buscar recursivamente la carpeta chest_xray
        for item in downloaded_path.rglob("*"):
            if item.is_dir() and item.name == "chest_xray":
                chest_xray_source = item
                break
        
        # Si no encuentra chest_xray, usar el directorio completo
        if chest_xray_source is None:
            chest_xray_source = downloaded_path
            print(f"⚠️ No se encontró carpeta 'chest_xray', usando: {chest_xray_source}")
        
        # Mover/copiar al directorio objetivo
        if final_dataset_path.exists():
            shutil.rmtree(final_dataset_path)
        
        print(f"📦 Copiando dataset a: {final_dataset_path}")
        shutil.copytree(chest_xray_source, final_dataset_path)
        
        # Verificar la estructura final
        print("\n📊 Verificando estructura del dataset:")
        verify_dataset_structure(final_dataset_path)
        
        print(f"\n✅ Dataset descargado exitosamente en: {final_dataset_path}")
        return str(final_dataset_path)
        
    except Exception as e:
        print(f"❌ Error descargando el dataset: {e}")
        print("\n💡 Soluciones posibles:")
        print("1. Verificar conexión a internet")
        print("2. Verificar autenticación de Kaggle")
        print("3. Descargar manualmente desde: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        return None

def verify_dataset_structure(dataset_path):
    """
    Verifica y muestra la estructura del dataset descargado.
    
    Args:
        dataset_path (str/Path): Ruta al dataset
    """
    dataset_path = Path(dataset_path)
    
    splits = ['train', 'test', 'val']
    classes = ['NORMAL', 'PNEUMONIA']
    total_images = 0
    
    print("\n" + "="*50)
    print("ESTRUCTURA DEL DATASET")
    print("="*50)
    
    for split in splits:
        split_path = dataset_path / split
        if split_path.exists():
            print(f"\n📁 {split.upper()}:")
            split_total = 0
            
            for class_name in classes:
                class_path = split_path / class_name
                if class_path.exists():
                    # Contar imágenes
                    images = list(class_path.glob('*.jpeg')) + \
                            list(class_path.glob('*.jpg')) + \
                            list(class_path.glob('*.png'))
                    count = len(images)
                    split_total += count
                    print(f"   {class_name:>9}: {count:>4} imágenes")
                else:
                    print(f"   {class_name:>9}: ❌ No encontrado")
            
            print(f"   {'TOTAL':>9}: {split_total:>4} imágenes")
            total_images += split_total
        else:
            print(f"\n📁 {split.upper()}: ❌ No encontrado")
    
    print(f"\n🎯 TOTAL GENERAL: {total_images} imágenes")
    print("="*50)

def get_dataset_info(dataset_path):
    """
    Obtiene información detallada del dataset.
    
    Args:
        dataset_path (str/Path): Ruta al dataset
    
    Returns:
        dict: Información del dataset
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        return None
    
    dataset_info = {
        'train': {'NORMAL': [], 'PNEUMONIA': []},
        'test': {'NORMAL': [], 'PNEUMONIA': []},
        'val': {'NORMAL': [], 'PNEUMONIA': []}
    }
    
    for split in ['train', 'test', 'val']:
        split_path = dataset_path / split
        if split_path.exists():
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_path = split_path / class_name
                if class_path.exists():
                    images = list(class_path.glob('*.jpeg')) + \
                            list(class_path.glob('*.jpg')) + \
                            list(class_path.glob('*.png'))
                    dataset_info[split][class_name] = images
    
    return dataset_info

if __name__ == "__main__":
    # Ejecutar descarga del dataset
    print("🚀 Iniciando descarga del dataset de radiografías de tórax...")
    
    dataset_path = download_chest_xray_dataset()
    
    if dataset_path:
        print(f"\n🎉 ¡Dataset listo para usar!")
        print(f"📍 Ubicación: {dataset_path}")
    else:
        print("\n❌ No se pudo descargar el dataset.")
        print("Por favor, descárgalo manualmente desde Kaggle.")