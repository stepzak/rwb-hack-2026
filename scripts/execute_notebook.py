"""Запуск всех ячеек текущим Python; сохраняет только успешно выполненный ноутбук."""
from pathlib import Path
import json, os, sys, tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    path=ROOT/'notebooks/final_solution.ipynb'
    nb=nbformat.read(path,as_version=4)
    # Отдельный временный kernelspec: пользовательские ядра не меняются.
    with tempfile.TemporaryDirectory(prefix='rwb-kernel-') as temp:
        kernel=Path(temp)/'rwb-final';kernel.mkdir()
        (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Python (RWB final)','language':'python'}))
        manager=KernelSpecManager(kernel_dirs=[temp])
        client=NotebookClient(nb,timeout=900,kernel_name='rwb-final',resources={'metadata':{'path':str(ROOT)}},allow_errors=False)
        client.km=client.create_kernel_manager();client.km.kernel_spec_manager=manager
        client.execute()
    nbformat.validate(nb)
    # Не публикуем локальный путь к интерпретатору из metadata виджетов.
    nb.metadata.pop('widgets',None)
    nbformat.write(nb,path)
    print(json.dumps({'executed_code_cells':sum(c.cell_type=='code' for c in nb.cells),'errors':0,'notebook':'notebooks/final_solution.ipynb'}))
