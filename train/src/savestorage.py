import dropbox
from dropbox.exceptions import ApiError, AuthError
from pathlib import Path
import sys

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
                try:
                    self.dbx.files_upload(fp_in.read(), fp_out)
                except ApiError as err:
                    if (err.error.is_path() and err.error.get_path().reason.is_insufficient_space()):
                        print("ERROR: Cannot back up; insufficient space.")
                    elif err.user_message_text:
                        print(err.user_message_text)
                    else:
                        print(err)

        else:
            try:
                self.dbx.files_upload(fp_in.read(), fp_out)
            except ApiError as err:
                    if (err.error.is_path() and err.error.get_path().reason.is_insufficient_space()):
                        print("ERROR: Cannot back up; insufficient space.")
                    elif err.user_message_text:
                        print(err.user_message_text)
                    else:
                        print(err)
            else:
                print(f"File {fp_in} saved at {fp_out}.")

        
    def show_folder(self, folder):
        for entry in dbx.files_list_folder(folder).entries:
            print(entry.name)
