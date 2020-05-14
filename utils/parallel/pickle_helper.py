def mark_static_module(module):
    """
    Some modules are static, which means they will remain
    the same whether you import it in process A or process
    B.
    If your module contains reference to functions, objects
    or anything inside a CDLL (usually the reference is a
    pointer), it is not pickable by dill, and will cause
    nasty errors, however, by marking this module as "Static",
    dill will recognize this module as a builtin module and
    not saving the states of this module, dill will only save
    a reference to it in this situation.
    """
    del module.__file__