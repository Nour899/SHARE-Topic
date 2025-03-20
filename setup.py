from setuptools import setup, find_packages

setup(
    name='SHARE_topic',  
    version='0.1.0', 
    author='Nour El Kazwini',  
    author_email='nelkazwi@sissa.com',  
    description='Topic modeling for ATAC and scRNA-seq data',  
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown',  
    url='https://github.com/Nour899/SHARE-Topic/', 
    packages=find_packages(),  
    install_requires=[
        'scanpy>=1.9.8', 
        'torch>=2.1.2',
        'torch-scatter>=2.1.1',
    ],
    classifiers=[
       
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6', 
    
)