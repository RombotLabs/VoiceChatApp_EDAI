import os


class DirectoryFilesFetcher:
    """
    Fetch all files in a directory recursively.
    """

    def __init__(self, directory_path):
        if not os.path.isdir(directory_path):
            raise ValueError(f"Invalid directory path: {directory_path}")

        self.directory_path = directory_path
        self.files = []

    def fetch_files(self):
        self.files.clear()

        for root, _, files in os.walk(self.directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                self.files.append(file_path)

        return self.files

    def get_files(self):
        return self.files

