#example here: 
#https://github.com/pypa/sampleproject/blob/db5806e0a3204034c51b1c00dde7d5eb3fa2532e/setup.py

from setuptools import setup, find_packages

setup(
    name='SHARE_topic',  # Required. Your package's name.
    version='0.1.0',  # Required. Your package's version.
    author='Nour El Kazwini',  # Optional. The package author's name.
    author_email='nelkazwi@sissa.com',  # Optional. The package author's email address.
    description='Topic modeling for ATAC and scRNA-seq data',  # Required. A short description of your package.
    long_description=open('README.md').read(),  # Optional. A long description of your package. This can be read from a file.
    long_description_content_type='text/markdown',  # Optional. Specifies the content type (markdown or rst) of the long description.
    url='https://github.com/Nour899/SHARE-Topic/',  # Optional. The URL to your package's repository or website.
    packages=find_packages(),  # Required. Automatically find all the packages and subpackages.
    install_requires=[
        'scanpy>=1.9.8', 
        'torch>=2.1.2',
        'torch-scatter>=2.1.1',
    ],
    classifiers=[
        # Optional. A list of classifiers that describe your package. Check the list of classifiers at https://pypi.org/classifiers/
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Optional. Specify which Python versions your package is compatible with.
    # Additional optional arguments can be specified here.
)