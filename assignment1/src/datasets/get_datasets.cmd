IF NOT EXIST "cifar-10-batches-py" (
    certutil -urlcache -split -f "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" cifar-10-python.tar.gz
    tar -xzvf cifar-10-python.tar.gz
    del cifar-10-python.tar.gz
)