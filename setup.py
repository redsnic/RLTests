import setuptools

setuptools.setup(
   name='RLTools',
   version='0.99',
   description='A small package with a collection of easy to use tools for RL. The implementations are minimal and simple.',
   author='Nicolo Rossi',
   author_email='nicolo.rossi@bsse.ethz.ch',
   install_requires=['wheel', 'torch', 'gymnasium'],
   packages=setuptools.find_packages()
)