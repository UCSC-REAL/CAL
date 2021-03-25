import os
import sys
import json
import argparse

import torch
from torch._six import PY3

import errno
import hashlib
import gzip
import tarfile
import zipfile
try:
    from tqdm.auto import tqdm  # automatically select proper tqdm submodule if available
except ImportError:
    try:
        from tqdm import tqdm
    except ImportError:
        # fake tqdm if it's not installed
        class tqdm(object):

            def __init__(self, total=None, disable=False,
                         unit=None, unit_scale=None, unit_divisor=None):
                self.total = total
                self.disable = disable
                self.n = 0
                # ignore unit, unit_scale, unit_divisor; they're just for real tqdm

            def update(self, n):
                if self.disable:
                    return

                self.n += n
                if self.total is None:
                    sys.stderr.write("\r{0:.1f} bytes".format(self.n))
                else:
                    sys.stderr.write("\r{0:.1f}%".format(100 * self.n / float(self.total)))
                sys.stderr.flush()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.disable:
                    return

                sys.stderr.write('\n')

# from .options import Options

def save_opt_to_json(opt, mpath):
    json_dir = os.path.join(mpath, "opt.json")
    argparse_dict = vars(opt)
    with open(json_dir, 'w') as outfile:
        json.dump(argparse_dict, outfile)
    print ("configs have been dumped into %s" % json_dir)

def load_json_as_argparse(mpath):
    try:
        json_dir = os.path.join(mpath, "opt.json")
        js = open(json_dir).read()
        data = json.loads(js)
        opt = argparse.Namespace()
        for key, val in data.items():
            setattr(opt, key, val) 
        return opt
    except Exception as e:
        print("No such file or directory %s" % (json_dir))
 
def dict2list(adict):
    alist = []
    for k, v in adict.items():
        if isinstance(v, list):
            val_list = [str(a) for a in v]
            alist += ([k] + val_list)
        else:
            alist += [k, str(v)]
    
    return alist

def get_parameters(options, param_dict, json_path):
    '''
    determing the method for loading parameters, either from:
        * command line
        * direct dictionary input
        * json file
    '''
    if json_path is None:
        if param_dict is None:
            opt = options.parse()
        else:
            if isinstance(param_dict, dict):
                opt = options.parse_args(dict2list(param_dict))
            else:
                raise ValueError("invalid inputs")
    else:
        if param_dict is not None:
            print("warning: conflict input methods, using the input dictionary")
            if isinstance(param_dict, dict):
                opt = options.parse_args(dict2list(param_dict))
            else:
                raise ValueError("invalid inputs")
        elif os.path.exists(json_path):
            opt = load_json_as_argparse(json_path)
        else:
            raise ValueError("invalid json file path")
    return opt

def expandabspath(optpath):
    r"""
    expand optpath to adspath so that can be used directly on second-level directory
                                                              ~~~~~~~~~~~~
    consider cases:
        optpath = ./datasets
        or = datasets
    """
    if not os.path.exists(optpath):
        pwd_list = os.getcwd().split("/")[1:-1] # get rid of "" and the last folder
        optfolder = optpath.split("/")[-1] #NOTE applicable only to format ./datasets or datasets
        abspath = os.path.join("/", *pwd_list, optfolder)
    else:
        abspath = optpath
    return abspath

# ===============================================================================================================
# below copied from pytorch official repo:
# https://github.com/pytorch/vision/blob/61763fa955ef74077a1d3e1aa5da36f7c606943a/torchvision/datasets/utils.py
# ===============================================================================================================
def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update

def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()

def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)

def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)

def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:   # download the file
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")

def _is_tarxz(filename):
    return filename.endswith(".tar.xz")

def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_tgz(filename):
    return filename.endswith(".tgz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")

def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path) and PY3:
        # .tar.xz archive only supported in Python 3.x
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)

def download_and_extract_archive(url, download_root, extract_root=None, filename=None,
                                 md5=None, remove_finished=False):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)

def iterable_to_str(iterable):
    return "'" + "', '".join([str(item) for item in iterable]) + "'"

def verify_str_arg(value, arg=None, valid_values=None, custom_msg=None):
    if not isinstance(value, torch._six.string_classes):
        if arg is None:
            msg = "Expected type str, but got type {type}."
        else:
            msg = "Expected type str for argument {arg}, but got type {type}."
        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)

    if valid_values is None:
        return value

    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = ("Unknown value '{value}' for argument {arg}. "
                   "Valid values are {{{valid_values}}}.")
            msg = msg.format(value=value, arg=arg,
                             valid_values=iterable_to_str(valid_values))
        raise ValueError(msg)

    return value