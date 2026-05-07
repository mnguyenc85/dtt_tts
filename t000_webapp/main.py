from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import time

app = FastAPI()

# Cấu hình static và templates
app.mount("/static", StaticFiles(directory="t000_webapp/static"), name="static")
templates = Jinja2Templates(directory="t000_webapp/templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=
            {"request": request, "audio_file": None}
        )