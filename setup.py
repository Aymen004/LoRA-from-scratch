from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="main",
    version="0.1.0",
    author="Aymen KHOMSI",  # Update with your name
    description="A PyTorch implementation of LoRA (Low-Rank Adaptation) built from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Aymen004/LoRA-from-scratch", 
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
    ],
)