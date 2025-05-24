from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello"}

@app.get("/docs")
def docs_redirect():
    return {"message": "Docs ready"}
