import os

class Index:
    root_dir = ''
    data_dir = ''

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def run_program(self):
        print('start script')

if __name__ == "__main__":
    index = Index(os.getcwd())
    index.run_program()
