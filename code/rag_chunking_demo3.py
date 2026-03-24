# 特殊语言切分
from langchain_text_splitters import (RecursiveCharacterTextSplitter, Language)

# 一段Python代码示例
python_code = """
import os
import sys

class MyClass:
    def __init__(self, name):
        self.name = name

    def say_hello(self):
        print(f"Hello, {self.name}!")

def my_function(x, y):
    return x + y

if __name__ == "__main__":
    instance = MyClass("World")
    instance.say_hello()
    result = my_function(5, 3)
    print(f"Result is {result}")
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=150, chunk_overlap=0)

# 进行分块
chunks = python_splitter.split_text(python_code)

# 我们来看看分块结果
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i + 1} ---")
    print(chunk)
    print(f"(长度: {len(chunk)})\n")
