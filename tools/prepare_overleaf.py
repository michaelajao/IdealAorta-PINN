"""Package the active manuscript for Overleaf; optionally compile the bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'doc/IdealAorta_EACFM'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'report/overleaf_review')
    parser.add_argument('--build', action='store_true', help='Compile from the copied bundle using latexmk/pdfLaTeX')
    parser.add_argument('--refresh', action='store_true', help='Refresh an unchanged prior generated bundle after verifying its manifest')
    args = parser.parse_args()
    destination = args.output.resolve()
    archive = destination.with_suffix('.zip')
    receipt_path = destination.parent / (destination.name + '_manifest.json')
    if destination.exists() or archive.exists() or receipt_path.exists():
        if not args.refresh or not all(p.exists() for p in (destination, archive, receipt_path)):
            parser.error('Output already exists; choose a fresh --output path or use --refresh on a generated bundle.')
        previous = json.loads(receipt_path.read_text(encoding='utf-8'))
        expected = previous['files']
        actual = {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}
        if actual != set(expected):
            parser.error('Existing bundle has added or missing files; choose a fresh output path.')
        with zipfile.ZipFile(archive) as stream:
            if set(stream.namelist()) != set(expected):
                parser.error('Existing zip changed; choose a fresh output path.')
            for name, record in expected.items():
                if (hashlib.sha256((destination / name).read_bytes()).hexdigest() != record['sha256'] or
                        hashlib.sha256(stream.read(name)).hexdigest() != record['sha256']):
                    parser.error('Existing bundle or zip was edited; choose a fresh output path.')
    destination.mkdir(parents=True, exist_ok=True)
    files = [SOURCE / 'main.tex']
    for directory, extensions in (('bib', {'.bib'}), ('tables', {'.tex'}),
                                   ('figures', {'.pdf', '.png', '.jpg', '.jpeg', '.dat'})):
        files.extend(sorted(p for p in (SOURCE / directory).rglob('*')
                            if p.is_file() and p.suffix.lower() in extensions))
    manifest = {}
    for path in files:
        relative = path.relative_to(SOURCE)
        copied = destination / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copied)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == digest
        manifest[relative.as_posix()] = {'bytes': path.stat().st_size, 'sha256': digest}
    pending_archive = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(pending_archive, 'w', compression=zipfile.ZIP_DEFLATED) as stream:
        for name in manifest:
            stream.write(destination / name, arcname=name)
    pending_archive.replace(archive)
    receipt = {'source_directory': str(SOURCE), 'upload_directory': str(destination),
               'archive': str(archive), 'main_document': 'main.tex', 'compiler': 'pdfLaTeX',
               'revision_colour': 'red; unchanged text black; scientific figure colours retained',
               'scope': 'coauthor review; unresolved scientific wording remains listed in doc/revision_audit_2026-10-23/audit_response.txt',
               'files': manifest}
    if args.build:
        build = destination.parent / (destination.name + '_build')
        build.mkdir(exist_ok=True)
        command = ['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                   '-outdir=' + str(build), 'main.tex']
        with (build / 'console.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=destination, stdout=log, stderr=subprocess.STDOUT)
        receipt['build_returncode'] = result.returncode
        receipt['built_pdf'] = str(build / 'main.pdf')
        if result.returncode:
            raise SystemExit('Build failed; inspect ' + str(build / 'console.log'))
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'files': len(manifest), 'directory': str(destination),
                      'zip': str(archive), 'manifest': str(receipt_path)}, indent=2))


if __name__ == '__main__':
    main()
