from setuptools import setup, find_packages

setup(
    name="german-text-analyzer",
    version="0.1.0",
    description="A lightweight NLP pipeline for German text processing",
    author="Shikhar Sharma",
    author_email="shik99990@gmail.com",
    url="https://github.com/shikjazz/german-text-analyzer",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "spacy>=3.5,<4.0",
        "transformers>=4.30,<5.0",
        "torch>=2.0",
        "nltk>=3.8",
        "click>=8.0",
        "rich>=13.0",
    ],
    entry_points={
        "console_scripts": [
            "german-text-analyzer=german_text_analyzer.__main__:cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Text Processing :: Linguistic",
        "Natural Language :: German",
    ],
)
