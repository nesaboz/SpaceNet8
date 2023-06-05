# pip install fabric2
# pip install parmiko

from fabric2 import Connection
import paramiko


GRU_IP = '184.105.175.58'
A6000_IP = '172.83.14.18'


def copy_files_using_fabric(remote_file_paths, local_file_paths, host_ip=A6000_IP, download=True, username='paperspace', pw=''):
    """
    Copy files to/from a remote server using SSH.
    
    Args:
        files (list(str)): _description_
        host_ip (str, optional): _description_. Defaults to ''.
        download (bool, optional): _description_. Defaults to True.
        username (str, optional): _description_. Defaults to 'paperspace'.
        pw (str, optional): _description_. Defaults to ''.
    """

    # Connect to the remote server
    c = Connection(host=host_ip, user=username, connect_kwargs={'password': pw})

    for remote_file_path, local_file_path in zip(remote_file_paths, local_file_paths):
        fcn = c.get if download else c.put
        fcn(local=local_file_path, remote=remote_file_path)
        
    # Close the connection
    c.close()


def copy_files_using_parmiko(file_paths, replace_path, host_ip='A6000_IP', download=True, username='paperspace', pw=''):
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(host_ip, username=username, password=pw)

        sftp = ssh.open_sftp()

        for file_path in file_paths:
            remote_path = file_path.replace(replace_path, '')
            local_path = file_path if download else remote_path

            if download:
                sftp.get(file_path, local_path)
                print(f"File '{file_path}' downloaded to '{local_path}'")
            else:
                sftp.put(file_path, remote_path)
                print(f"File '{file_path}' uploaded to '{remote_path}'")

        sftp.close()
        ssh.close()

    except paramiko.AuthenticationException:
        print("Authentication failed. Please check the credentials.")
    except paramiko.SSHException as ssh_exception:
        print(f"Error connecting to the host: {str(ssh_exception)}")
    except FileNotFoundError:
        print("One or more files were not found.")


if __name__=='__main__':

    
    remote_root = 'tmp/share/runs/spacenet8/nenad'
    local_root = '/Users/nenad.bozinovic/Downloads'

    remote_file_paths = []

    local_file_paths = [x.replace(remote_root, local_root) for x in remote_file_paths]
    
    copy_files_using_fabric(remote_file_paths, local_file_paths)



