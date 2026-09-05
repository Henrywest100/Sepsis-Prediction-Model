# CRUD - PPGD
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Todo(BaseModel):
    title: str

@app.get('/')
def homes():
    return {"message": "Hello World!"}


@app.get('/todos')
def get_todos():
    return[
        {'id': 1, 'title': 'Learn APIs'},
        {'id': 2, 'title': 'Building FastAPI project'},
        {'id': 3, 'title': 'Learn MLOPS'},
        {'id': 4, 'title': 'Deploy Elvara Sepsis Model'},
    ]


@app.post('/todos')
def create_todo(todo: Todo):
    return todo


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

