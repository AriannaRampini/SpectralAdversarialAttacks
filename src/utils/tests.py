
import os 
import datetime

def create_timestamped_folder(str_folder: str) -> str:
    r"""Simply creates a folder to save the tests upput data.
        Returns folder name: str_folder + timestamp.
     """
    timestamp = datetime.datetime.now().timestamp()
    readable  = datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d_%H:%M:%S')
    str_out   = str_folder + readable
    
    path = os.path.realpath(str_out)
    os.makedirs(path)
    return path
