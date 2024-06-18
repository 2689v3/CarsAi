from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn

from loguru import logger

from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from predict4 import CarsAi


class AiSchema(BaseModel):
    content: UploadFile = File(...)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/ai', response_class=FileResponse)
async def recive_file(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")
    content = await file.read()
    with open("image.jpg", "wb") as f:
        f.write(content)
    logger.info(f"Saved file: {file.filename}")
    logger.info(f"Ai started")
    CarsAi()
    logger.info(f"Ai finished")
    return FileResponse("head_output.jpg")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
