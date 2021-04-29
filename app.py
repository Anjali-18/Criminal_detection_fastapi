import uvicorn
from fastapi import FastAPI, File, UploadFile, Response
from pydantic import BaseModel
import face_rec as frc
import numpy as np
import cv2
from starlette.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image 

cfd = FastAPI()

origins = ["*"]

cfd.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# @cfd.get("/")
# def home():
#     return "Welcome to criminal face detection API"

@cfd.get("/impro")
def impro():
    return ''

@cfd.post("/impro")
async def upload_image(file: UploadFile=File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    matched_image = frc.classify_face(img)
    #print("hii")
    img = cv2.imread('matched_image.png')
    print(matched_image)

    # return frc.classify_face(img)
    # return Response("matched_image.png", media_type='application/octet-stream',filename="matched_image.png",)
    return FileResponse("matched_image.png", media_type='application/octet-stream',filename="matched_image.png")


if __name__ == "__app__":
    uvicorn.run(app, debug=True, port=8000)
