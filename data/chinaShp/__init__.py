import os
from mycode._Class import Paths

dir_data = os.path.abspath(os.path.dirname(__file__))

data_paths = Paths.get_data_paths(dir_data)

if __name__ == '__main__':
    data_paths.show()













