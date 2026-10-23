"""Read-only API adapter for the flat local and nested corrected Brosnan layouts."""
import importlib
import importlib.util


def api_module(name):
    nested=importlib.util.find_spec('idealaorta_pinn.reconstruction') is not None
    if name=='data':
        path='idealaorta_pinn.data.full_export' if nested else 'idealaorta_pinn.data'
    elif name=='numerics':
        path='idealaorta_pinn.reconstruction.numerics' if nested else 'idealaorta_pinn.postprocess'
    else:
        path=('idealaorta_pinn.reconstruction.' if nested else 'idealaorta_pinn.')+name
    return importlib.import_module(path)


def constants():
    name='idealaorta_pinn.reconstruction' if importlib.util.find_spec('idealaorta_pinn.reconstruction') else 'idealaorta_pinn.config'
    module=importlib.import_module(name)
    return float(module.RHO),float(module.MU)
