from setuptools import setup

setup(
    name="micloc",
    packages=["micloc"],
    description="python module for simulating multi-mic localization.",
    version="0.1",
    url="https://gitlab.com/synsense/research/auditoryprocessing/multi-mic-snn-localization",
    author="Saeid Haghighatshoar",
    author_email="saeid.haghighatshoar@synsense.ai",
    download_url="https://gitlab.com/synsense/research/auditoryprocessing/multi-mic-snn-localization",
    keywords=[
        "multi-mic localization",
        "array processing",
        "audio applications",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "rockpool",
        "tqdm",
        "matplotlib",
        "cvxpy",
    ],
)
