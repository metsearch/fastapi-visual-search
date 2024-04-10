import io
import base64
import uvicorn

from PIL import Image
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from search import get_similar_images
from utilities.utils import *

similar_images = pull_images('data/')[:15]

class APIServer:
    def __init__(self, host:str, port:int):
        self.app = FastAPI()
        self.host = host
        self.port = port
        self.templates = Jinja2Templates(directory='templates')
        self.app.mount('/static', StaticFiles(directory='static'), name='static')
        
        self.app.add_api_route('/', self.home, methods=['GET'])
        # self.app.add_api_route('/vectorize', self.vectorize, methods=['POST'])
        self.app.add_api_route('/upload', self.upload, methods=['POST'])
    
    def start_service(self):
        self.config = uvicorn.Config(app=self.app, host=self.host, port=self.port)
        self.server = uvicorn.Server(config=self.config)
        self.server.run()

    async def home(self, request: Request):
        return self.templates.TemplateResponse('home.html', {'request': request})
    
    # async def vectorize(self, request: Request):
    #     return self.templates.TemplateResponse('home.html', {'request': request})
    
    async def upload(self, request: Request, file: UploadFile = File(...)):
        try:
            contents = file.file.read()
            query_image = Image.open(io.BytesIO(contents))
            similar_images = get_similar_images(query_image)
            # encoded_similar_images = [base64.b64encode(image) for image in similar_images]
            encoded_similar_images = []
            for image in similar_images:
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                encoded_similar_images.append(encoded_image)
            encoded_query_image = base64.b64encode(contents).decode('utf-8')
            return self.templates.TemplateResponse('upload.html', {'request': request,  'queryImage': encoded_query_image, 'similarImages': encoded_similar_images})
        except Exception as e:
            logger.error(f'Exception: {e}')
            logger.exception(e)
            return {'message': 'There was an error uploading the file'}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            logger.error(f'Exception: {exc_type} {exc_value}')
            logger.exception(traceback)

if __name__ == '__main__':
    with APIServer(host='0.0.0.0', port=8000) as server:
        server.start_service()