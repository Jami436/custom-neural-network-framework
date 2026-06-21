from setuptools import setup, find_packages

setup(
    name="my_framework",
    version="0.1.0",
    author="Muhammad Jami Ahad",
    author_index="Jami436",
    description="A modular, object-oriented Deep Learning framework built from scratch using NumPy",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/Jami436/custom-neural-network-framework",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
)