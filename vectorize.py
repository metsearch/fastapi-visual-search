import faiss

import torch
from PIL import Image

from sentence_transformers import SentenceTransformer

from utilities.utils import *
    
def build_index(path2data='data/', path2index='data/images.index', path2cache='cache/', index_dim=512, batch_size=32, model_name='clip-ViT-B-32'):
    if not os.path.exists(path2cache):
        os.makedirs(path2cache)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    index = faiss.IndexFlatL2(index_dim)
    
    model = SentenceTransformer(
        model_name_or_path=model_name,
        cache_folder=path2cache,
        device = device
    )
    
    image_paths = pull_images(path2data)
    image_paths = sorted(image_paths)
    
    nb_images = len(image_paths)
    logger.info(f'Number of images: {nb_images}')
    
    for cursor in range(0, nb_images, batch_size):
        sample = image_paths[cursor:cursor+batch_size]
        image_accumulator = []
        for image_path in sample:
            with Image.open(fp=image_path) as fp:
                image_copy = fp.copy()
                image_accumulator.append(image_copy)
        
        image_embeddings = model.encode(
            sentences=image_accumulator,
            batch_size=batch_size,
            show_progress_bar=True
        )
        
        index.add(image_embeddings)
        # binary_image_embeddings = quantize_embedding(image_embeddings)
        
    faiss.write_index(index, path2index)
    logger.info(f'Index saved to {path2index}')

if __name__ == '__main__':
    logger.info('Vectorizing...')
    build_index()