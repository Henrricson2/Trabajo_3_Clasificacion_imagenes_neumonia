"""
Funciones para preprocesamiento de imágenes médicas.
"""

import cv2
import numpy as np
from skimage import exposure, filters, morphology, segmentation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def resize_image(image, target_size=(224, 224), interpolation=cv2.INTER_AREA):
    """
    Redimensiona una imagen al tamaño objetivo.
    
    Args:
        image (np.ndarray): Imagen de entrada
        target_size (tuple): Tamaño objetivo (width, height)
        interpolation: Método de interpolación de OpenCV
    
    Returns:
        np.ndarray: Imagen redimensionada
    """
    return cv2.resize(image, target_size, interpolation=interpolation)

def apply_clahe(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Aplica Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Especialmente útil para imágenes médicas.
    
    Args:
        image (np.ndarray): Imagen en escala de grises
        clip_limit (float): Límite de contraste
        tile_grid_size (tuple): Tamaño de la grilla de tiles
    
    Returns:
        np.ndarray: Imagen con CLAHE aplicado
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image.astype(np.uint8))

def normalize_intensity(image, method='minmax', target_range=(0, 1)):
    """
    Normaliza la intensidad de una imagen.
    
    Args:
        image (np.ndarray): Imagen de entrada
        method (str): Método de normalización ('minmax', 'zscore', 'percentile')
        target_range (tuple): Rango objetivo para normalización minmax
    
    Returns:
        np.ndarray: Imagen normalizada
    """
    image_float = image.astype(np.float32)
    
    if method == 'minmax':
        min_val, max_val = image_float.min(), image_float.max()
        if max_val > min_val:
            normalized = (image_float - min_val) / (max_val - min_val)
            normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
        else:
            normalized = image_float
    
    elif method == 'zscore':
        mean_val, std_val = image_float.mean(), image_float.std()
        if std_val > 0:
            normalized = (image_float - mean_val) / std_val
        else:
            normalized = image_float
    
    elif method == 'percentile':
        p1, p99 = np.percentile(image_float, [1, 99])
        if p99 > p1:
            normalized = np.clip((image_float - p1) / (p99 - p1), 0, 1)
            normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
        else:
            normalized = image_float
    
    else:
        raise ValueError(f"Método desconocido: {method}")
    
    return normalized

def remove_noise(image, method='gaussian', **kwargs):
    """
    Elimina ruido de una imagen.
    
    Args:
        image (np.ndarray): Imagen de entrada
        method (str): Método de filtrado ('gaussian', 'median', 'bilateral')
        **kwargs: Argumentos adicionales para el método específico
    
    Returns:
        np.ndarray: Imagen filtrada
    """
    if method == 'gaussian':
        kernel_size = kwargs.get('kernel_size', 5)
        sigma = kwargs.get('sigma', 1.0)
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    elif method == 'median':
        kernel_size = kwargs.get('kernel_size', 5)
        return cv2.medianBlur(image.astype(np.uint8), kernel_size)
    
    elif method == 'bilateral':
        d = kwargs.get('d', 9)
        sigma_color = kwargs.get('sigma_color', 75)
        sigma_space = kwargs.get('sigma_space', 75)
        return cv2.bilateralFilter(image.astype(np.uint8), d, sigma_color, sigma_space)
    
    else:
        raise ValueError(f"Método desconocido: {method}")

def enhance_edges(image, method='unsharp_mask', **kwargs):
    """
    Realza los bordes de una imagen.
    
    Args:
        image (np.ndarray): Imagen de entrada
        method (str): Método de realce ('unsharp_mask', 'laplacian')
        **kwargs: Argumentos adicionales
    
    Returns:
        np.ndarray: Imagen con bordes realzados
    """
    if method == 'unsharp_mask':
        radius = kwargs.get('radius', 1.0)
        amount = kwargs.get('amount', 1.0)
        return exposure.adjust_gamma(
            filters.unsharp_mask(image, radius=radius, amount=amount)
        )
    
    elif method == 'laplacian':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced = cv2.filter2D(image.astype(np.float32), -1, kernel)
        return np.clip(enhanced, 0, 1)
    
    else:
        raise ValueError(f"Método desconocido: {method}")

def segment_lung_region(image, method='threshold', **kwargs):
    """
    Segmenta la región pulmonar en radiografías de tórax.
    
    Args:
        image (np.ndarray): Imagen de radiografía
        method (str): Método de segmentación ('threshold', 'watershed')
        **kwargs: Argumentos adicionales
    
    Returns:
        np.ndarray: Máscara binaria de la región pulmonar
    """
    if method == 'threshold':
        # Aplicar filtro gaussiano para suavizar
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Umbralización adaptiva
        threshold_value = kwargs.get('threshold', 0.1)
        if image.max() <= 1.0:
            threshold_value = threshold_value
        else:
            threshold_value = threshold_value * 255
        
        _, binary = cv2.threshold(blurred.astype(np.uint8), 
                                 threshold_value, 255, cv2.THRESH_BINARY)
        
        # Operaciones morfológicas para limpiar la máscara
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Encontrar el contorno más grande (presumiblemente los pulmones)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.fillPoly(mask, [largest_contour], 255)
        
        return mask
    
    else:
        raise ValueError(f"Método desconocido: {method}")

def apply_mask(image, mask):
    """
    Aplica una máscara a una imagen.
    
    Args:
        image (np.ndarray): Imagen original
        mask (np.ndarray): Máscara binaria
    
    Returns:
        np.ndarray: Imagen con máscara aplicada
    """
    if mask.max() > 1:
        mask = mask / 255.0
    
    return image * mask

def preprocess_pipeline(image, target_size=(224, 224), apply_clahe=True, 
                       normalize=True, remove_noise_flag=False, enhance_edges_flag=False,
                       segment_lungs=False):
    """
    Pipeline completo de preprocesamiento para imágenes médicas.
    
    Args:
        image (np.ndarray): Imagen original
        target_size (tuple): Tamaño objetivo
        apply_clahe (bool): Si aplicar CLAHE
        normalize (bool): Si normalizar intensidades
        remove_noise_flag (bool): Si aplicar filtro de ruido
        enhance_edges_flag (bool): Si realzar bordes
        segment_lungs (bool): Si segmentar región pulmonar
    
    Returns:
        dict: Diccionario con imagen procesada y máscaras/información adicional
    """
    result = {'original': image.copy()}
    
    # 1. Redimensionar
    processed = resize_image(image, target_size)
    result['resized'] = processed.copy()
    
    # 2. Eliminar ruido (opcional)
    if remove_noise_flag:
        processed = remove_noise(processed, method='gaussian')
        result['denoised'] = processed.copy()
    
    # 3. Aplicar CLAHE
    if apply_clahe:
        if processed.dtype != np.uint8:
            processed_uint8 = (processed * 255).astype(np.uint8)
        else:
            processed_uint8 = processed
        
        processed = apply_clahe(processed_uint8).astype(np.float32) / 255.0
        result['clahe'] = processed.copy()
    
    # 4. Segmentación de pulmones (opcional)
    lung_mask = None
    if segment_lungs:
        lung_mask = segment_lung_region(processed)
        processed = apply_mask(processed, lung_mask)
        result['lung_mask'] = lung_mask
        result['segmented'] = processed.copy()
    
    # 5. Realzar bordes (opcional)
    if enhance_edges_flag:
        processed = enhance_edges(processed)
        result['enhanced'] = processed.copy()
    
    # 6. Normalización final
    if normalize:
        processed = normalize_intensity(processed, method='minmax', target_range=(0, 1))
        result['normalized'] = processed.copy()
    
    result['final'] = processed
    result['lung_mask'] = lung_mask
    
    return result

def batch_preprocess(images, **preprocessing_params):
    """
    Procesa un lote de imágenes aplicando el pipeline de preprocesamiento.
    
    Args:
        images (list): Lista de imágenes a procesar
        **preprocessing_params: Parámetros para el pipeline de preprocesamiento
    
    Returns:
        np.ndarray: Array con imágenes procesadas
    """
    processed_images = []
    
    for image in images:
        result = preprocess_pipeline(image, **preprocessing_params)
        processed_images.append(result['final'])
    
    return np.array(processed_images)