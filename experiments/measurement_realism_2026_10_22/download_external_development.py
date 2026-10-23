"""Fetch ONLY the declared 4DFlowNet adapter-development geometry.

No evaluation-file argument is provided. Interrupted transfers retain their
partial file; a new process must not be launched while the existing handle is
live. A completed file is reusable only after its size and MD5 are verified.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import shutil
import time
import urllib.request

NAME='aorta01_HR.h5'
URL='https://ndownloader.figshare.com/files/42875266'
SIZE=4161195048
MD5='10043cda23d60e7a3e06f519fa1e6ac9'


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)


def download(destination):
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    lock=root/'aorta01_download.lock'
    with lock.open('x',encoding='utf-8') as stream:
        json.dump({'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},stream)
    receipt={'role':'adapter development only','url':URL,'name':NAME,'expected_bytes':SIZE,
             'expected_md5':MD5,'evaluation_arrays_accessed':False,
             'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    progress=root/'aorta01_download.json';final=root/NAME;partial=root/(NAME+'.part')
    try:
        if final.exists() and partial.exists():raise ValueError('retain and resolve duplicate final/partial data before continuing')
        file=final if final.exists() else partial
        count=file.stat().st_size if file.exists() else 0
        if count>SIZE:raise ValueError('existing file exceeds the declared release size')
        if not final.exists() and shutil.disk_usage(root).free<SIZE-count+1024**3:
            raise OSError('insufficient space for the declared file plus 1 GiB reserve')
        md5=hashlib.md5();sha=hashlib.sha256()
        if count:
            with file.open('rb') as source:
                for block in iter(lambda:source.read(8*1024**2),b''):md5.update(block);sha.update(block)
        receipt.update(status='downloading',bytes=count,resumed_from_bytes=count);write(progress,receipt)
        if not final.exists() and count<SIZE:
            request=urllib.request.Request(URL,headers={'Range':f'bytes={count}-'}) if count else urllib.request.Request(URL)
            with urllib.request.urlopen(request,timeout=60) as response,partial.open('ab') as output:
                if count and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {count}-')):
                    raise ValueError('server did not honour the exact resume range')
                updated=time.monotonic()
                for block in iter(lambda:response.read(4*1024**2),b''):
                    if count+len(block)>SIZE:raise ValueError('response exceeds the declared release size')
                    output.write(block);md5.update(block);sha.update(block);count+=len(block)
                    if time.monotonic()-updated>5:
                        receipt.update(bytes=count);write(progress,receipt)
                        print({'bytes':count,'total':SIZE},flush=True);updated=time.monotonic()
        if count!=SIZE or md5.hexdigest()!=MD5:raise ValueError('download size or release MD5 does not match')
        if not final.exists():partial.replace(final)
        receipt.update(status='verified_download',bytes=count,md5=md5.hexdigest(),sha256=sha.hexdigest(),
                       finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(progress,receipt);return receipt
    except BaseException as exc:
        receipt.update(status='download_failed; retain partial file',error=repr(exc),bytes=partial.stat().st_size if partial.exists() else 0)
        write(progress,receipt);raise
    finally:lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(download(parser.parse_args().output),indent=2))
