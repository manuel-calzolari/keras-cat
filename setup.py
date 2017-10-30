from setuptools import setup
from setuptools import find_packages

setup(name='keras-cat',
      version='0.1',
      description='Keras model for categorical features support in neural networks',
      url='https://github.com/manuel-calzolari/keras-cat',
      author='Manuel Calzolari',
      author_email='',
      license='GPLv3',
      install_requires=['Keras>=2.0.0'],
      packages=find_packages())
