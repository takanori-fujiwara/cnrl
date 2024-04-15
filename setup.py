from distutils.core import setup

setup(
    name="cnrl",
    version=0.01,
    packages=[""],
    package_dir={"": "."},
    install_requires=["numpy", "scikit-learn"],
    py_modules=["cnrl"],
)
