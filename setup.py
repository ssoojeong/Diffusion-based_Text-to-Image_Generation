from setuptools import setup


setup(
    name="diffusion",
    packages=[
        "model/model_t2i",
        "model/model_t2i.clip",
        "model/model_t2i.tokenizer",
    ],
    package_data={
        "model/model_t2i.tokenizer": [
            "bpe_simple_vocab_16e6.txt.gz",
            "encoder.json.gz",
            "vocab.bpe.gz",
        ],
        "model/model_t2i.clip": ["config.yaml"],
    },
    install_requires=[
        "Pillow",
        "attrs",
        "torch",
        "filelock",
        "requests",
        "tqdm",
        "ftfy",
        "regex",
        "numpy",
    ],
    author="OpenAI",
)
