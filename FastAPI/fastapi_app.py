import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"Hello": "World"}


# Handling HTTP methods
@app.get("/request")
def request():
    return {"Hello": "GET"}

@app.post("/request")
def request_post():
    return {"Hello": "POST"}


# URL Parameters
@app.get("/employee/{id}")
def get_employee(id: int):
    return {"id": id}


# Serving Static Files
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

templates = Jinja2Templates(directory='templates')

@app.get("/welcome", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", reload=True)