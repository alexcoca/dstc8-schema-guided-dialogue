from setuptools import find_packages, setup


def readme():
    with open('README.md') as f:
        return f.read()


INSTALL_REQUIRES = [
          'numpy>=1.20.2, <2.0.0',
          'typing-extensions>=3.7.4.3',
]

DESCRIPTION = 'Clone of https://github.com/google-research-datasets/dstc8-schema-guided-dialogue with additional ' \
              'data analysis utilities.'

setup(name='sgd_utils',
      author='Alexandru Coca',
      author_email='ac2123@cam.ac.uk',
      version='0.1',
      description=DESCRIPTION,
      url='https://github.com/alexcoca/dstc8-schema-guided-dialogue',
      packages=find_packages(exclude=['tests']),
      include_package_data=True,
      python_requires='>=3.8',
      install_requires=INSTALL_REQUIRES,
      test_suite='tests',
      zip_safe=False)
