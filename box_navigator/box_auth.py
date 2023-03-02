"""Helper Classes to locally recreate Box filesystem structure

Note: The DEVELOPER_TOKEN_60MINS must be refreshed every 60 minutes.
Go to https://salesforcecorp.app.box.com/developers/console/app/1366340/configuration
And create a developer token. Then copy it into this variable, below. 

Example. Replicate Box filesystem locally:
    bn = BoxNavigator()
    bn.locally_recreate_filesystem_directory_structure(root_path="./box_data")
    bn.maybe_download_filesystem(root_path="./box_data")
Note: If the token times out, you'll need to restart this. Don't worry,
it won't re-download anything that is already there.
"""

from boxsdk import Client, OAuth2
import os
import time
from tqdm import tqdm

DEVELOPER_TOKEN_60MINS="XXXX"

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

class BoxNavigator:
    def __init__(self, token=DEVELOPER_TOKEN_60MINS):
        # Create Authentication Client
        auth = OAuth2(
            client_id='XXXX',
            client_secret='XXXX',
            access_token=token,
        )
        client = Client(auth)
        self.client = client

        # Set Initial State to Root Dir
        self.curr_dir = '0'
        self.root_dir = '0'

        # Print Opening Message
        me = client.user().get()
        print("Initializing client")
        print('My user ID is {0}'.format(me.id))

        # Parse filesystem structure
        self.filesystem = self._parse_file_structure()



    def locally_recreate_filesystem_directory_structure(self, root_path="./"):
        folders = self._filesystem_folders()
        mkdir(root_path)
        for folder in folders:
            path = os.path.join(root_path, folder)
            mkdir(path)


    def _filesystem_folders(self):
        """Returns a list of the fullpaths to the folders of this filesystem"""
        folders = [item[0] for item in self.filesystem if item[2] == 'folder']
        return folders


    def _parse_file_structure(self):
        """Equivalent to os.walk.

        Recurses through the Box Account and creates the following:
            [[/path, item.id, item.type],
             [/path/to/folder, item.id, item.type],
             [/path/to/item, item.id, item.type],
             ...
            ]

        Returns:
            The list of lists, above.
        """
        time.sleep(1)
        filesystem = []
        def recurse(self, dir_id, dir_path, filesystem):
            items = self.client.folder(folder_id=dir_id).get_items()
            for item in items:
                item_path = os.path.join(dir_path, item.name)
                filesystem.append([item_path,
                                   item.id,
                                   item.type])
                if item.type.lower() == "folder":
                    print("Parsing {}".format(item_path))
                    new_path = os.path.join(dir_path, item.name)
                    recurse(self, item.id, new_path, filesystem)

        recurse(self, self.root_dir, '', filesystem)
        return filesystem


    def ls(self, dir=None):
        if dir is None:
            dir = self.curr_dir
        items = self.client.folder(folder_id=dir).get_items()
        for item in items:
            print('{0}:{1}:{2}'.format(item.type.capitalize(), item.id, item.name))


    def cd(self, dir=None):
        if dir is None:
            self.curr_dir = self.root_dir
        else:
            self.curr_dir = dir


    def maybe_download_filesystem(self, root_path):
        """Download any files that don't currently exist.
        Mirror the Box filesystem, at root_dir
        """
        self.locally_recreate_filesystem_directory_structure(root_path)
        for item in tqdm(self.filesystem):
            filepath = os.path.join(root_path, item[0])
            self.maybe_download_file(item[1], filepath)


    def maybe_download_file(self, fileid, filepath):
        """If filepath DNE, download fileid to filepath location.

        Args:
            fileid(str): a number, the fileid
            filepath(str): format of "./path/to/a/file"
        """
        file = self.client.file(fileid)
        if os.path.exists(filepath):
            print("Skipping (exists): {}".format(filepath))
            return
        print("Downloading: ID {} to {}".format(fileid, filepath))
        with open(filepath, 'wb') as open_file:
            file.download_to(open_file)
            open_file.close()


    def download_file(self, fileid, dir='.'):
        """Downloads file given by fileid to dir/file_name.

        Args:
            fileid(str): the file id
            dir(str): dir on local machine to download ot
        """
        # get filename from fileid
        file = self.client.file(fileid)
        filename = file.get().name

        # join dir and filename -> filename
        local_path = os.path.join(dir, filename)

        # open filestream with the given filename.
        with open(local_path, 'wb') as open_file:
            file.download_to(open_file)
            open_file.close()
            print("Downloading {}:{} -> {}".format(fileid,filename, local_path))
        # write to file










