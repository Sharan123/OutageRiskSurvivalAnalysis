import warnings

# PyG 2.7 warns on import that its `torch_geometric.distributed` module is deprecated; the study
# never uses it. Silence exactly that notice (every entry point imports this package first).
warnings.filterwarnings("ignore", message=r"`torch_geometric\.distributed` has been deprecated",
                        category=DeprecationWarning)
