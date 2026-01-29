from setuptools import setup, find_packages

setup(
    name='TechNowNet',
    version='0.1.0',
    url='https://github.com/lee-ju/TechNowNet.git',
    author='KISTI',
    description='TechNowNet: A Contemporary Multi-Domain Knowledge Semantic Network through Cross-Domain Embedding Fusion.',
    packages=find_packages(),
    install_requires=[
        'git+https://github.com/lee-ju/TechNowNet',
    ],
)
