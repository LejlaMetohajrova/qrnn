import tensorflow as tf
import os


if __name__ == "__main__":
    if not os.path.exists('aclImdb'):
        os.system('./download.sh')
