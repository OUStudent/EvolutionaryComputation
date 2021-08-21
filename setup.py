import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE/"README.md").read_text()

setup(name='EvolutionaryComputation',
      version='0.0.4',
      description='Python module containing algorithms in the domain of Evolutionary Computation',
      long_description=README,
      long_description_content_type="text/markdown",
      author='Brandon Morgan',
      author_email="morganscottbrandon@gmail.com",
      license="MIT",
      install_requires=['matplotlib',
                        'numpy']
      )