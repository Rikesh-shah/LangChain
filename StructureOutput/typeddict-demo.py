from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person : Person = {'name':'Rikesh', 'age': 25}

print(type(new_person))