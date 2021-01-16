import dropbox
from pathlib import Path

class DropboxLoader:
    def __init__(self, token, access_key):
        self.token = token
        self.key = access_key
        self.dbx = None

    def log_on(self):
        self.dbx = dropbox.Dropbox(self.token)
        user_info = self.dbx.users_get_current_account()
        print(f"Successfully logged on Dropbox as {user_info.name.display_name}")

    def save_file(self, fp_in, fp_out):
        if type(fp_in) is str or isinstance(fp_in, Path):
            with open(fp_in, 'rb') as f:
                self.dbx.files_upload(fp_in.read(), fp_out)
        else:
            try:
                self.dbx.files_upload(fp_in.read(), fp_out)
            except:
                print("Error: must be an _io.TextIOWrapper.")
        print(f"File {fp_in} saved at {fp_out}.")

    def show_folder(self, folder):
        for entry in dbx.files_list_folder(folder).entries:
            print(entry.name)
